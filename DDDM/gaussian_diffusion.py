"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import ph_loss,pl_loss


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps,beta_start,beta_end):
    """
    scheduler for VP model
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        
        beta_start = beta_start / num_diffusion_timesteps
        beta_end = beta_end / num_diffusion_timesteps
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_sigma_schedule(num_diffusion_timesteps,sigma_min,sigma_max):

    """
    scheduler for VE model
    """

    return np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), num_diffusion_timesteps))






class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss

    PH = enum.auto() # Pseudo-Huber
    PL = enum.auto() # Pseudo-LPIPS

   


class VP_Diffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    
    """

    def __init__(
        self,
        betas,
        loss_type,
        c = 0.0,
    ):
       
        self.loss_type = loss_type
        self.c = c

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

      

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )


    def p_sample(self, model, x_T, T, x_bar,model_kwargs=None):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        
        sample = x_T - model(x_T, T, context=x_bar,**model_kwargs)
        sample = sample.clamp(-1, 1)
        return sample

       

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        condition=None,
        model_kwargs=None,
        device=None,
        sample_steps=1,
        
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
       
        x_T = th.randn(*shape, device=device)
        x_bar = th.randn(*shape, device=device)
        T = th.tensor([self.num_timesteps] * shape[0], device=device)

        for i in range(sample_steps):
            with th.no_grad():
                out = self.p_sample(
                    model,
                    x_T,
                    x_bar,
                    T,
                    model_kwargs=model_kwargs,
                )
                x_bar = out
               
        return x_bar

   

    def training_losses(self, model, x_start, t, index,condition,model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
        model_output = model(x_t, t, context=condition,**model_kwargs)
        model_output = x_t-model_output

        
        if self.loss_type == LossType.MSE:
            
                                 
            terms["guide"] = mean_flat((model_output-x_start) ** 2)
            terms["iter"] = mean_flat((model_output-condition) ** 2)
            

            #model.module.update_xbar(model_output,index)
        
        elif self.loss_type == LossType.PL:

            terms["guide"]= pl_loss(model_output,x_start,self.c)
            terms["iter"] = pl_loss(model_output,condition,self.c)
        
        elif self.loss_type == LossType.PH:

            terms["guide"]= ph_loss(model_output,x_start,self.c)
            terms["iter"]= ph_loss(model_output,condition,self.c)


        else:
            raise NotImplementedError(self.loss_type)

        return terms, model_output


class VE_Diffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    
    """

    def __init__(
        self,
        sigmas,
        loss_type,
        c = 0.0,
    ):
       
        self.loss_type = loss_type
        self.c = c

        # Use float64 for accuracy.
        sigmas = np.array(sigmas, dtype=np.float64)
        self.sigmas = sigmas
        assert len(sigmas.shape) == 1, "sigmas must be 1-D"
        #assert (sigmas > 0).all() and (sigmas <= 1).all()
        assert (sigmas > 0).all()

        self.num_timesteps = int(sigmas.shape[0])
        self.sigma_min = sigmas[0]
        self.sigma_max = sigmas[-1]

      

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            x_start
            + _extract_into_tensor(self.sigmas, t, x_start.shape)**2
            * noise
        )


    def p_sample(self, model, x_T, T, x_bar,model_kwargs=None):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        kappa = self.sigma_min / self.sigma_max
        sample = kappa * x_T + (1-kappa) * model(x_T, T, context=x_bar,**model_kwargs)
        sample = sample.clamp(-1, 1)
        return sample

       

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        condition=None,
        model_kwargs=None,
        device=None,
        sample_steps=1,
        
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
       
        x_T = th.randn(*shape, device=device)
        x_bar = th.randn(*shape, device=device)
        T = th.tensor([self.num_timesteps] * shape[0], device=device)

        for i in range(sample_steps):
            with th.no_grad():
                out = self.p_sample(
                    model,
                    x_T,
                    x_bar,
                    T,
                    model_kwargs=model_kwargs,
                )
                x_bar = out
               
        return x_bar

   

    def training_losses(self, model, x_start, t, index,condition,model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        sigma_t = _extract_into_tensor(self.sigmas, t, x_start.shape)
        kappa = self.sigma_min/sigma_t

        terms = {}
        
        model_output = model(x_t, t, context=condition,**model_kwargs)
        model_output = kappa*x_t + (1-kappa)*model_output

        
        if self.loss_type == LossType.MSE:
            
            terms["guide"] = mean_flat((model_output-x_start) ** 2)
            terms["iter"] = mean_flat((model_output-condition) ** 2)
            

            #model.module.update_xbar(model_output,index)
        
        elif self.loss_type == LossType.PL:
            
            
            terms["guide"]= pl_loss(model_output,x_start,self.c)
            terms["iter"] = pl_loss(model_output,condition,self.c)
        
        elif self.loss_type == LossType.PH:

                
            terms["guide"]= ph_loss(model_output,x_start,self.c)
            terms["iter"]= ph_loss(model_output,condition,self.c)


        else:
            raise NotImplementedError(self.loss_type)

        return terms, model_output



def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
