import reverb
from tf_agents.specs import tensor_spec
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils


def new_replay_buffer(data_spec, capcity):
    table_name = "uniform_table"
    replay_buffer_signature = tensor_spec.from_spec(data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)
    table = reverb.Table(
        table_name,
        max_size=capcity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature,
    )

    reverb_server = reverb.Server([table])

    return reverb_replay_buffer.ReverbReplayBuffer(
        data_spec,
        table_name=table_name,
        sequence_length=None,
        local_server=reverb_server,
    )


def new_reverb_observer(data_spec, capcity, buffer, table_name="uniform_table"):
    rb_observer = reverb_utils.ReverbAddEpisodeObserver(
        buffer.py_client, table_name, capcity
    )

    return rb_observer
