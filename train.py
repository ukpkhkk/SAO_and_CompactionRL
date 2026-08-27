import gc

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action


def train(args):
    configure_logger()
    release_train = args.release_train

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        K = getattr(args, 'critic_update_ratio', 2)
        critic_handle = None
        handle = None

        if args.use_critic:
            # value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            # if actor_trains:
            #     ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))

            # else:
            #     ray.get(value_refs)

            if actor_trains:
                # ============================================================
                # 正常模式：Actor 更新 1 次，Critic 更新 K 次
                # ============================================================
                
                # ----- 第 1 次 Critic 更新（计算 values，供 Actor 使用） -----
                critic_handle = critic_model.async_train(rollout_id, rollout_data_ref)
                ray.get(critic_handle)  # 等待 Critic 完成，确保 values 已写入 rollout_data
                
                # ----- Actor 更新 1 次 -----
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=critic_handle))
                
                # ----- 后续 K-1 次 Critic 更新（复用同一批数据） -----
                for update_idx in range(K - 1):
                    handle = critic_model.async_train(rollout_id, rollout_data_ref)
                    ray.get(handle)
            else:
                # ============================================================
                # 预热模式：只更新 Critic 1 次，不更新 Actor
                # ============================================================
                critic_handle = critic_model.async_train(rollout_id, rollout_data_ref)
                ray.get(critic_handle)

        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        # Drop driver-side Ray ObjectRefs as soon as the trainers have consumed
        # them. Otherwise the previous rollout batch can stay pinned in Ray's
        # object store until the next assignment overwrites these variables.
        rollout_data_ref = None
        critic_handle = None
        handle = None
        gc.collect()

        if release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            force_sync = release_train or rollout_id == args.num_rollout - 1
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        offload_train(actor_trains)
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
