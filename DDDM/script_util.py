import argparse
import inspect

from . import gaussian_diffusion as gd
from .unet import UNetModel

NUM_CLASSES = 10


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=32,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        concat=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        sigma_min=0.01,
        sigma_max=50,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        use_ph = False,
        use_pl = False,
        use_CA = True,
        VP = True,
        c = 1.4e-4,
    )


def create_model_and_diffusion(
    image_size,
    class_cond,
    concat,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    use_checkpoint,
    use_scale_shift_norm,
    dropout,
    sigma_min,
    sigma_max,
    diffusion_steps,
    noise_schedule,
    VP,
    use_pl,
    use_ph,
    use_CA,
    c
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        concat=concat,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        use_CA = use_CA,
    )
    diffusion = create_gaussian_diffusion(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
        VP = VP,
        use_pl=use_pl,
        use_ph=use_ph,
        c=c,
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    concat,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    use_CA,
):
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        in_channels=(3 if not concat else 6),
        model_channels=num_channels,
        out_channels=3 ,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        use_CA = use_CA
    )



def create_gaussian_diffusion(
    sigma_min,
    sigma_max,
    steps=1000,
    VP=True,
    noise_schedule="linear",
    use_pl = False,
    use_ph = True,
    c = 0.0,
):

    
    
    if use_ph:
        loss_type = gd.LossType.PH
    elif use_pl:
        
        loss_type = gd.LossType.PL
    else:
        loss_type = gd.LossType.MSE
    
    if VP:
        betas = gd.get_named_beta_schedule(noise_schedule, steps,sigma_min,sigma_max)
        return gd.VP_Diffusion(betas,loss_type,c)
    else:
        sigmas = gd.get_sigma_schedule(steps,sigma_min,sigma_max)
        return gd.VE_Diffusion(steps,loss_type,c)
    


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
