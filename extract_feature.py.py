import torch
from dinov3.models import vision_transformer as vits  # 假设路径这样
from PIL import Image
import torch
from torchvision.transforms import v2
from omegaconf import OmegaConf


def get_img():
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def build_model(args, img_size=224, device=None):
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            pos_embed_rope_base=args.pos_embed_rope_base,
            pos_embed_rope_min_period=args.pos_embed_rope_min_period,
            pos_embed_rope_max_period=args.pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=args.pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=args.pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=args.pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=args.pos_embed_rope_rescale_coords,
            qkv_bias=args.qkv_bias,
            layerscale_init=args.layerscale,
            norm_layer=args.norm_layer,
            ffn_layer=args.ffn_layer,
            ffn_bias=args.ffn_bias,
            proj_bias=args.proj_bias,
            n_storage_tokens=args.n_storage_tokens,
            mask_k_bias=args.mask_k_bias,
            untie_cls_and_patch_norms=args.untie_cls_and_patch_norms,
            untie_global_and_local_cls_norm=args.untie_global_and_local_cls_norm,
            device=device,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)

        return teacher
        


cfg = OmegaConf.load("/data/tanyuyi/code/dinov3-frozen-path/results/UNI/config.yaml")
model=build_model(
        cfg.student,
        device="meta",
    )
model.to_empty(device="cuda")
pretrained_weights="/data/tanyuyi/code/dinov3-frozen-path/results/UNI/eval/training_12499/teacher_checkpoint.pth"
state_dict = torch.load(pretrained_weights, map_location="cpu")["teacher"]
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

msg = model.load_state_dict(state_dict, strict=True)
model.eval()



img_size = 224
img = get_img()
transform = make_transform(img_size)

with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        batch_img = batch_img.cuda()
        depths = model(batch_img)#torch.Size([1, 1024])



print(depths)
print(depths.shape)