from collections.abc import Iterable, MutableMapping, Sequence
from typing import Any, TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from .model import ImageTextModel
from .utils import unwrap_model

try:
    from torch.distributed import nn as dist_nn
    from torch import distributed as dist

    supports_distributed = True
except ImportError:
    dist_nn = Any if TYPE_CHECKING else None
    dist = Any if TYPE_CHECKING else None
    supports_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = Any if TYPE_CHECKING else None


def _dist_all_gather(tensor: torch.Tensor, group: dist.ProcessGroup | None = None,
                     async_op: bool = False) -> Sequence[torch.Tensor]:
    gathered_tensors = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(gathered_tensors, tensor, group=group, async_op=async_op)
    return gathered_tensors


def gather_features(x: torch.Tensor, local_loss: bool = False, gather_with_grad: bool = False,
                    use_horovod: bool = False,
                    return_as_tensor: bool = True) -> torch.Tensor | Sequence[torch.Tensor]:
    assert supports_distributed, ("`torch.distributed` did not import correctly, please use a PyTorch version with"
                                  " support for it.")
    assert dist.is_initialized()

    if use_horovod:
        assert hvd, "Please install horovod."

        with torch.set_grad_enabled(gather_with_grad):
            # FIXME: not sure if `hvd.allgather_object` supports gradients in the same way as `hvd.allgather`.
            gathered = (hvd.allgather if return_as_tensor else hvd.allgather_object)(x)
    else:
        gathered = (dist_nn.all_gather if gather_with_grad else _dist_all_gather)(x)

    if not gather_with_grad and not local_loss:
        # Ensure grads for the local rank. For local loss, we don't need it because the original tensor is going to be
        # used as well, which preserves the grads.

        if isinstance(gathered, torch.Tensor):
            gathered = list(gathered.chunk(dist.get_world_size()))

        gathered[dist.get_rank()] = x

    return gathered if not return_as_tensor or isinstance(gathered, torch.Tensor) else torch.cat(gathered)


class ClipLoss(nn.Module):
    def __init__(
            self,
            model: ImageTextModel,
            do_gather: bool = True,
            local_loss: bool = False,
            gather_with_grad: bool = False,
            cache_labels: bool = False,
            use_horovod: bool = False,
            name: str = "contrastive_loss",
    ) -> None:
        super().__init__()
        self.model = model
        self.do_gather = do_gather
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.use_horovod = use_horovod
        self.name = name

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def _get_ground_truth(self, device: torch.device, num_logits: int) -> torch.Tensor:
        if self.prev_num_logits == num_logits and device in self.labels:
            labels = self.labels[device]
        else:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if dist.is_initialized() and dist.get_world_size() > 1 and self.local_loss:
                labels += num_logits * dist.get_rank()
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits

        return labels

    def _get_logit_computation_args(self, image_features: torch.Tensor,
                                    text_features: torch.Tensor) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        if self.do_gather and dist.is_initialized() and dist.get_world_size() > 1:
            all_image_features = gather_features(image_features, self.local_loss, self.gather_with_grad,
                                                 self.use_horovod)
            all_text_features_list = gather_features(text_features, self.local_loss, self.gather_with_grad,
                                                     self.use_horovod, return_as_tensor=False)

            # There may be extra negative text features.
            # In this case, we put first together (in order) all text features that have a corresponding image feature;
            # then, the rest.
            all_text_features = torch.cat([t[:len(image_features)] for t in all_text_features_list]
                                          + [t[len(image_features):] for t in all_text_features_list])

            if self.local_loss:
                logits_per_image_args = image_features, all_text_features
                logits_per_text_args = text_features, all_image_features
            else:
                logits_per_image_args = all_image_features, all_text_features
                logits_per_text_args = all_text_features, all_image_features
        else:
            logits_per_image_args = image_features, text_features
            logits_per_text_args = text_features, image_features

        # Consider the case that there are extra negative text features, which we ignore here:
        logits_per_text_args = (logits_per_text_args[0][:len(logits_per_image_args[0])], logits_per_text_args[1])

        return logits_per_image_args, logits_per_text_args

    def _get_logits(self, image_features: torch.Tensor, text_features: torch.Tensor) -> Iterable[torch.Tensor]:
        return (unwrap_model(self.model).compute_similarity(*logits)
                for logits in self._get_logit_computation_args(image_features, text_features))

    def _compute_loss_one_way(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor,
                **kwargs) -> MutableMapping[str, torch.Tensor]:
        logits_list = list(self._get_logits(image_features, text_features))
        labels = self._get_ground_truth(image_features.device, len(logits_list[0]))
        return {self.name: sum(self._compute_loss_one_way(logits, labels) for logits in logits_list) / len(logits_list)}


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            model: ImageTextModel,
            caption_loss_weight: float,
            clip_loss_weight: float,
            pad_id: int = 0,  # pad_token for open_clip custom tokenizer
            do_gather: bool = True,
            local_loss: bool = False,
            gather_with_grad: bool = False,
            cache_labels: bool = False,
            use_horovod: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            do_gather=do_gather,
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            use_horovod=use_horovod,
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, logits: torch.Tensor,  # noqa
                labels: torch.Tensor, **kwargs) -> MutableMapping[str, torch.Tensor]:
        if self.clip_loss_weight:
            clip_loss = self.clip_loss_weight * super().forward(image_features, text_features)[self.name]
        else:
            clip_loss = torch.tensor(0)

        caption_loss = self.caption_loss_weight * self.caption_loss(logits.permute(0, 2, 1), labels)

        return {self.name: clip_loss, "caption_loss": caption_loss}


class DistillClipLoss(ClipLoss):
    @staticmethod
    def _dist_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(  # noqa
            self,
            image_features: torch.Tensor,
            text_features: torch.Tensor,
            dist_image_features: torch.Tensor,
            dist_text_features: torch.Tensor,
            **kwargs,
    ) -> MutableMapping[str, torch.Tensor]:
        logits_per_image, logits_per_text = self._get_logits(image_features, text_features)

        # FIXME: We suppose both models use the same function to compute the similarity, otherwise I'm not sure what's
        #  the correct way to compute the logits for the distilled model.
        dist_logits_per_image, dist_logits_per_text = self._get_logits(dist_image_features, dist_text_features)

        labels = self._get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (self._compute_loss_one_way(logits_per_image, labels)
                            + self._compute_loss_one_way(logits_per_text, labels)) / 2

        distill_loss = (self._dist_loss(dist_logits_per_image, logits_per_image) +
                        self._dist_loss(dist_logits_per_text, logits_per_text)) / 2

        return {self.name: contrastive_loss, "distill_loss": distill_loss}


def neighbour_exchange(from_rank: int, to_rank: int, tensor: torch.Tensor,
                       group: dist.ProcessGroup | None = None) -> torch.Tensor:
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    for req in torch.distributed.batch_isend_irecv([send_op, recv_op]):
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank: int, right_rank: int, tensor_to_left: torch.Tensor,
                             tensor_to_right: torch.Tensor,
                             group: dist.ProcessGroup | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank: int, to_rank: int, group: dist.ProcessGroup | None,  # noqa
                tensor: torch.Tensor) -> torch.Tensor:
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[None, None, None, torch.Tensor]:  # noqa
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank: int, to_rank: int, tensor: torch.Tensor,
                                 group: dist.ProcessGroup | None = None) -> torch.Tensor:
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank: int, right_rank: int, group: dist.ProcessGroup | None,  # noqa
                tensor_to_left: torch.Tensor, tensor_to_right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> tuple[None, None, None, torch.Tensor, torch.Tensor]:
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank: int, right_rank: int, tensor_to_left: torch.Tensor,
                                       tensor_to_right: torch.Tensor,
                                       group: dist.ProcessGroup | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """

    def __init__(
            self,
            model: ImageTextModel,
            do_gather: bool = True,
            bidir: bool = True,
            use_horovod: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.do_gather = do_gather
        self.use_horovod = use_horovod
        self.bidir = bidir

        assert not use_horovod  # FIXME: need to look at hvd ops for ring transfers

    @staticmethod
    def get_ground_truth(device: torch.device, dtype: torch.dtype, shape: tuple[int, int],
                         negative_only: bool = False) -> torch.Tensor:
        labels = -torch.ones(shape, device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(*shape, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features: torch.Tensor, text_features: torch.Tensor,
                   logit_bias: float | None = None) -> torch.Tensor:
        logits = unwrap_model(self.model).compute_similarity(image_features, text_features)
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features: torch.Tensor, text_features: torch.Tensor, logit_bias: float | None = None,
              negative_only: bool = False) -> torch.Tensor:
        logits = self.get_logits(image_features, text_features, logit_bias)
        labels = self.get_ground_truth(image_features.device, image_features.dtype, logits.shape,
                                       negative_only=negative_only)
        return -F.logsigmoid(labels * logits).sum() / image_features.shape[0]

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, logit_bias: float | None = None,
                **_kwargs) -> MutableMapping[str, torch.Tensor]:  # noqa
        loss = self._loss(image_features, text_features, logit_bias)

        if self.do_gather and dist.is_initialized() and dist.get_world_size() > 1:
            # exchange text features w/ neighbor world_size - 1 times
            right_rank = (dist.get_rank() + 1) % dist.get_world_size()
            left_rank = (dist.get_rank() - 1 + dist.get_world_size()) % dist.get_world_size()
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(dist.get_world_size() - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(dist.get_world_size() - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss}
