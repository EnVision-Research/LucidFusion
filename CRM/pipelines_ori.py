import torch
from libs.base_utils import do_resize_content
from imagedream.ldm.util import (
    instantiate_from_config,
    get_obj_from_str,
)
from omegaconf import OmegaConf
from PIL import Image
import PIL
import rembg
class TwoStagePipeline(object):
    def __init__(
        self,
        stage1_model_config,
        stage2_model_config,
        stage1_sampler_config,
        stage2_sampler_config,
        device="cuda",
        dtype=torch.float16,
        resize_rate=1,
    ) -> None:
        """
        only for two stage generate process.
        - the first stage was condition on single pixel image, gererate multi-view pixel image, based on the v2pp config
        - the second stage was condition on multiview pixel image generated by the first stage, generate the final image, based on the stage2-test config
        """
        self.resize_rate = resize_rate

        self.stage1_model = instantiate_from_config(OmegaConf.load(stage1_model_config.config).model)
        self.stage1_model.load_state_dict(torch.load(stage1_model_config.resume, map_location="cpu"), strict=False)
        self.stage1_model = self.stage1_model.to(device).to(dtype)

        self.stage2_model = instantiate_from_config(OmegaConf.load(stage2_model_config.config).model)
        sd = torch.load(stage2_model_config.resume, map_location="cpu")
        self.stage2_model.load_state_dict(sd, strict=False)
        self.stage2_model = self.stage2_model.to(device).to(dtype)

        self.stage1_model.device = device
        self.stage2_model.device = device
        self.device = device
        self.dtype = dtype
        self.stage1_sampler = get_obj_from_str(stage1_sampler_config.target)(
            self.stage1_model, device=device, dtype=dtype, **stage1_sampler_config.params
        )
        self.stage2_sampler = get_obj_from_str(stage2_sampler_config.target)(
            self.stage2_model, device=device, dtype=dtype, **stage2_sampler_config.params
        )

    def stage1_sample(
        self,
        pixel_img,
        prompt="3D assets",
        neg_texts="uniform low no texture ugly, boring, bad anatomy, blurry, pixelated,  obscure, unnatural colors, poor lighting, dull, and unclear.",
        step=50,
        scale=5,
        ddim_eta=0.0,
    ):
        if type(pixel_img) == str:
            pixel_img = Image.open(pixel_img)

        if isinstance(pixel_img, Image.Image):
            if pixel_img.mode == "RGBA":
                background = Image.new('RGBA', pixel_img.size, (0, 0, 0, 0))
                pixel_img = Image.alpha_composite(background, pixel_img).convert("RGB")
            else:
                pixel_img = pixel_img.convert("RGB")
        else:
            raise
        uc = self.stage1_sampler.model.get_learned_conditioning([neg_texts]).to(self.device)
        stage1_images = self.stage1_sampler.i2i(
            self.stage1_sampler.model,
            self.stage1_sampler.size,
            prompt,
            uc=uc,
            sampler=self.stage1_sampler.sampler,
            ip=pixel_img,
            step=step,
            scale=scale,
            batch_size=self.stage1_sampler.batch_size,
            ddim_eta=ddim_eta,
            dtype=self.stage1_sampler.dtype,
            device=self.stage1_sampler.device,
            camera=self.stage1_sampler.camera,
            num_frames=self.stage1_sampler.num_frames,
            pixel_control=(self.stage1_sampler.mode == "pixel"),
            transform=self.stage1_sampler.image_transform,
            offset_noise=self.stage1_sampler.offset_noise,
        )

        stage1_images = [Image.fromarray(img) for img in stage1_images]
        # import pdb; pdb.set_trace()
        stage1_images.pop(self.stage1_sampler.ref_position)
        return stage1_images

    def stage2_sample(self, pixel_img, stage1_images, scale=5, step=50):
        if type(pixel_img) == str:
            pixel_img = Image.open(pixel_img)

        if isinstance(pixel_img, Image.Image):
            if pixel_img.mode == "RGBA":
                background = Image.new('RGBA', pixel_img.size, (0, 0, 0, 0))
                pixel_img = Image.alpha_composite(background, pixel_img).convert("RGB")
            else:
                pixel_img = pixel_img.convert("RGB")
        else:
            raise
        stage2_images = self.stage2_sampler.i2iStage2(
            self.stage2_sampler.model,
            self.stage2_sampler.size,
            "3D assets",
            self.stage2_sampler.uc,
            self.stage2_sampler.sampler,
            pixel_images=stage1_images,
            ip=pixel_img,
            step=step,
            scale=scale,
            batch_size=self.stage2_sampler.batch_size,
            ddim_eta=0.0,
            dtype=self.stage2_sampler.dtype,
            device=self.stage2_sampler.device,
            camera=self.stage2_sampler.camera,
            num_frames=self.stage2_sampler.num_frames,
            pixel_control=(self.stage2_sampler.mode == "pixel"),
            transform=self.stage2_sampler.image_transform,
            offset_noise=self.stage2_sampler.offset_noise,
        )
        stage2_images = [Image.fromarray(img) for img in stage2_images]
        return stage2_images

    def set_seed(self, seed):
        self.stage1_sampler.seed = seed
        self.stage2_sampler.seed = seed

    def __call__(self, pixel_img, prompt="3D assets", scale=5, step=50):
        pixel_img = do_resize_content(pixel_img, self.resize_rate)
        stage1_images = self.stage1_sample(pixel_img, prompt, scale=scale, step=step)
        stage2_images = self.stage2_sample(pixel_img, stage1_images, scale=scale, step=step)

        return {
            "ref_img": pixel_img,
            "stage1_images": stage1_images,
            "stage2_images": stage2_images,
        }

rembg_session = rembg.new_session()

def expand_to_square(image, bg_color=(0, 0, 0, 0)):
    # expand image to 1:1
    width, height = image.size
    if width == height:
        return image
    new_size = (max(width, height), max(width, height))
    new_image = Image.new("RGBA", new_size, bg_color)
    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
    new_image.paste(image, paste_position)
    return new_image

def remove_background(
    image: PIL.Image.Image,
    rembg_session = None,
    force: bool = False,
    **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        # explain why current do not rm bg
        print("alhpa channl not enpty, skip remove background, using alpha channel as mask")
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image

def do_resize_content(original_image: Image, scale_rate):
    # resize image content wile retain the original image size
    if scale_rate != 1:
        # Calculate the new size after rescaling
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        # Resize the image while maintaining the aspect ratio
        resized_image = original_image.resize(new_size)
        # Create a new image with the original size and black background
        padded_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        paste_position = ((original_image.width - resized_image.width) // 2, (original_image.height - resized_image.height) // 2)
        padded_image.paste(resized_image, paste_position)
        return padded_image
    else:
        return original_image

def add_background(image, bg_color=(255, 255, 255)):
    # given an RGBA image, alpha channel is used as mask to add background color
    background = Image.new("RGBA", image.size, bg_color)
    return Image.alpha_composite(background, image)


def preprocess_image(image, background_choice, foreground_ratio, backgroud_color):
    """
    input image is a pil image in RGBA, return RGB image
    """
    print(background_choice)
    if background_choice == "Alpha as mask":
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
    else:
        image = remove_background(image, rembg_session, force_remove=True)
    image = do_resize_content(image, foreground_ratio)
    image = expand_to_square(image)
    image = add_background(image, backgroud_color)
    return image.convert("RGB")



