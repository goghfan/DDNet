import argparse
import datetime
import torch
import wandb

from torch.utils.data import DataLoader
from torchvision import datasets
import sys
from ddpm import script_utils
import os
import data

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import time

def main():
    args = create_argparser().parse_args()
    device = args.device

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint,map_location=torch.device('cuda:0')))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint,map_location=torch.device('cuda:0')))

        if args.log_to_wandb:
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")

            run = wandb.init(
                project=args.project_name,
                # entity='treaptofun',
                config=vars(args),

                name=args.run_name,
            )
            wandb.watch(diffusion)

        batch_size = args.batch_size

        train_dataset = data.create_dataset(dataroot='data/dealed/',type='nii.gz',trainortest='train')
        test_dataset=data.create_dataset(dataroot='data/dealed/',type='nii.gz',trainortest='test')
        train_loader=script_utils.cycle(data.create_dataloader(train_dataset,batch_size=args.batch_size,phase='train'))
        test_loader=data.create_dataloader(test_dataset,batch_size=args.batch_size,phase='test')

        acc_train_loss = 0

        for iteration in range(1, args.iterations + 1):
            time_start=time.time()
            diffusion.train()

            x = next(train_loader)
            # x = (x.to(device)).unsqueeze(0)
            # y = (y.to(device)).unsqueeze(0)
            x = x.type(torch.cuda.FloatTensor)

            if args.use_labels:
                loss = diffusion(x, y)
            else:
                loss = diffusion(x)

            acc_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diffusion.module.update_ema()
            
            if iteration % args.log_rate == 0:
                # test_loss = 0
                # with torch.no_grad():
                #     diffusion.eval()
                #     for x, y in test_loader:
                #         x = x.to(device)
                #         y = y.to(device)

                #         if args.use_labels:
                #             loss = diffusion(x, y)
                #         else:
                #             loss = diffusion(x)

                #         test_loss += loss.item()
                
                # if args.use_labels:
                #     samples = diffusion.sample(10, device, y=torch.arange(10, device=device))
                # else:
                #     samples = diffusion.sample(10, device)
                
                # samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()

                # test_loss /= len(test_loader)
                acc_train_loss /= args.log_rate

                wandb.log({
                    # "test_loss": test_loss,
                    "train_loss": acc_train_loss,
                    # "samples": [wandb.Image(sample) for sample in samples],
                })

                acc_train_loss = 0
            time_end=time.time()
            print('This is the {} epoch || time = {} || loss = {}'.format(iteration+131000,time_end-time_start,acc_train_loss))
            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration+131000}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration+131000}-optim.pth"
                try: 
                    torch.save(diffusion.state_dict(), model_filename)
                    torch.save(optimizer.state_dict(), optim_filename)
                except FileNotFoundError:
                    os.makedirs(f"{args.log_dir}/")
                    torch.save(diffusion.state_dict(), model_filename)
                    torch.save(optimizer.state_dict(), optim_filename)
        if args.log_to_wandb:
            run.finish()
    except KeyboardInterrupt:
        if args.log_to_wandb:
            run.finish()
        print("Keyboard interrupt, run finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # gpus=[0,1,2,3]

    #device='cpu'
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    model_checkpoint = '~\\ddpm_logs\\4.1\\name-ddpm-2023-04-01-17-14-iteration-131000-model.pth' 
    optim_checkpoint = '~\\ddpm_logs\\4.1\\name-ddpm-2023-04-01-17-14-iteration-131000-optim.pth'
    defaults = dict(
        learning_rate=2e-4,
        batch_size=1,
        iterations=800000,

        log_to_wandb=True,
        log_rate=100,
        checkpoint_rate=1000,
        log_dir="DDPM/~/ddpm_logs",
        project_name='name',
        run_name=run_name,

        model_checkpoint=model_checkpoint,
        optim_checkpoint=optim_checkpoint,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
    )
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()