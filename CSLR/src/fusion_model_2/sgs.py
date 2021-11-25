import torch


def create_random_mask(l, p):
    mask = torch.zeros(l).bool()
    idx = torch.randperm(l)
    idx = idx[: int(l * p)]
    mask[idx] = True
    return mask


def create_batch_random_mask(ls, p, pooled=True):
    if pooled:
        return create_random_mask(sum(ls), p)
    return torch.cat([create_random_mask(l, p) for l in ls])


def create_sgs_applier(p_detach, lengths, pooled=True):
    """
    This allows to sample once but be applied on mulitple modules & features.
    """
    detached = create_batch_random_mask(lengths, p_detach, pooled)
    attached = ~detached

    def sgs_apply(module, *data):
        n = len(data[0])

        attaching = attached.any()
        detaching = detached.any()

        assert attaching or detaching

        if attaching:
            attached_output = module(*[d[attached] for d in data])

        if detaching:
            with torch.no_grad():
                detached_output = module(*[d[detached] for d in data])

        if attaching:
            slot = torch.empty(
                n, *attached_output.shape[1:], dtype=attached_output.dtype
            )
        else:
            slot = torch.empty(
                n, *detached_output.shape[1:], dtype=detached_output.dtype
            )

        slot = slot.to(data[0].device)

        if attaching:
            slot[attached] = attached_output

        if detaching:
            slot[detached] = detached_output

        return slot

    return sgs_apply
