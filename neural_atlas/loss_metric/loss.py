import torch


class Loss(torch.nn.Module):
    @staticmethod
    def distortion(
        batch_mapped_metric_tensor,                                             # (batch.size, num_charts, batch.chart_uv_sample_size, 2, 2)
        is_valid,                                                               # (batch_size, num_charts, batch_chart_uv_sample_size)
        distortion_eps
    ):
        eye = torch.eye(                                                        # (2, 2)
            2,
            dtype=batch_mapped_metric_tensor.dtype,
            device=batch_mapped_metric_tensor.device
        )
        conditioned_metric_tensor = (                                           # (batch.size, num_charts, batch.chart_uv_sample_size, 2, 2)
            batch_mapped_metric_tensor + distortion_eps * eye
        )

        # compute the forward & backward Dirichlet energies
        """
        NOTE:
            1. Absolute value of the conditioned metric tensor determinant is
               necessary to prevent negative backward Dirichlet energy leading
               to NaN scaled symmetric Dirichlet energy, as a result of
               numerical stability issues.
            2. `Loss.det_two_by_two()` is used instead of `torch.det()` due to
               random illegal memory access runtime errors.
        """
        conditioned_metric_tensor_trace = Loss.trace(conditioned_metric_tensor) # (batch.size, num_charts, batch.chart_uv_sample_size)
        forward_dirichlet_energy = conditioned_metric_tensor_trace              # (batch.size, num_charts, batch.chart_uv_sample_size)
        # the backward Dirichlet energy is essentially 
        # `Loss.trace(conditioned_metric_tensor.inverse())`
        backward_dirichlet_energy = conditioned_metric_tensor_trace \
                                    / torch.abs(Loss.det_two_by_two(
                                        conditioned_metric_tensor
                                    ))                                          # (batch.size, num_charts, batch.chart_uv_sample_size)

        # compute the mean forward & backward Dirichlet energies associated to
        # mapped tangent vectors that are valid, for each sample in the batch
        num_target_pcl_mapped_nn = is_valid.sum(dim=(1, 2))                     # (batch.size)
        forward_dirichlet_energy = torch.sum(                                   # (batch.size)
            forward_dirichlet_energy * is_valid, dim=(1, 2)
        ) / num_target_pcl_mapped_nn
        backward_dirichlet_energy = torch.sum(                                  # (batch.size)
            backward_dirichlet_energy * is_valid, dim=(1, 2)
        ) / num_target_pcl_mapped_nn

        # infer the global scale that yields the optimal scaled symmetric
        # Dirichlet energy, for each sample in the batch
        """
        NOTE:
            Absolute value of the squared optimal scale is necessary to prevent
            negative values leading to NaN optimal scale, as a result of
            numerical stability issues.
        """
        optimal_sim_scale = torch.sqrt(torch.abs(                               # (batch.size)
            (forward_dirichlet_energy / backward_dirichlet_energy).sqrt()
            - distortion_eps
        ))

        # compute the mean scaled symmetric Dirichlet energy, for each sample
        scaled_symmetric_dirichlet_energy = 2 * torch.sqrt(                     # (batch.size)
            forward_dirichlet_energy * backward_dirichlet_energy
        )

        # compute the scaled symmetric Dirichlet energy averaged across all 
        # samples, with the minimum offset value of 4 subtracted
        MIN_SCALED_SYMMETRIC_DIRICHLET_ENERGY = 4.0
        mean_scaled_symmetric_dirichlet_energy = (
            scaled_symmetric_dirichlet_energy.mean()
            - MIN_SCALED_SYMMETRIC_DIRICHLET_ENERGY
        )

        return mean_scaled_symmetric_dirichlet_energy, optimal_sim_scale        # (), (batch.size)

    @staticmethod
    def trace(tensor):
        """Batch compute the trace of a tensor
        Args:
            tensor (torch.Tensor): Tensor of shape (..., N, N)
        Returns:
            trace (torch.Tensor): Trace of shape (...)
        """
        return tensor.diagonal(dim1=-1, dim2=-2).sum(dim=-1)

    @staticmethod
    def det_two_by_two(tensor):
        """Batch compute the determinant of a tensor of 2x2 matrices
        Args:
            tensor (torch.Tensor): Tensor of shape (..., 2, 2)
        Returns:
            det_two_by_two (torch.Tensor): Determinant of shape (...)
        """
        return (
            tensor[..., 0, 0] * tensor[..., 1, 1]
            - tensor[..., 0, 1] * tensor[..., 1, 0]
        )
