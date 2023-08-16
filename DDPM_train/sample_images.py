import argparse
import torch
import torchvision

import SimpleITK as sitk
import sys
import ddpm.script_utils as script_utils
import glob
import os
from datetime import datetime
def main():
    args = create_argparser().parse_args()
    device = args.device
    path = 'D:\\Desktop\\diffusion model code\\DDPM\\~\\ddpm_logs_t2\\'
    model_paths= glob.glob(os.path.join(path,"*-{}".format('model.pth')))
    for path in model_paths:
        args.model_path=path
        model_index = path[path.find('-iteration-')+11:path.find('-model')]
        try:                
            diffusion = script_utils.get_diffusion_from_args(args).to(device)
            diffusion.load_state_dict(torch.load(args.model_path,map_location='cuda:0'))

            if args.use_labels:
                for label in range(10):
                    y = torch.ones(args.num_images // 10, dtype=torch.long, device=device) * label
                    samples = diffusion.sample(args.num_images // 10, device, y=y)

                    for image_id in range(len(samples)):
                        image = ((samples[image_id] + 1) / 2).clip(0, 1)
                        image = sitk.GetImageFromArray(image)
                        image = sitk.WriteImage(image,f"{args.save_dir}/{label}-{image_id+model_index}.nii.gz")
            else:
                dat = str(datetime.today())[-6:]
                samples = diffusion.module.sample(args.num_images, device)
                for image_id in range(len(samples)):
                    image = samples[image_id][0]
                    image_sample = sitk.ReadImage('data\\T2\\T2_MNI_and_resize\\OAS30001_MR_d0129_T2w.nii.gz')
                    image = sitk.GetImageFromArray(image)
                    image.SetDirection(image_sample.GetDirection())
                    image.SetOrigin(image_sample.GetOrigin())
                    image.SetSpacing(image_sample.GetSpacing())
                    sitk.WriteImage(image,f"{args.save_dir}/{model_index+dat}.nii.gz")

        except KeyboardInterrupt:
            print("Keyboard interrupt, generation finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(num_images=1, device=device)
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument('--schedule_low',type=float,default=9e-4)
    parser.add_argument('--schedule_high',type=float,default=0.01)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str,default='result/4.16/')
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()