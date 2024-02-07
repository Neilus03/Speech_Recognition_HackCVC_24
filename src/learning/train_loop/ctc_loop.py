import torch
from tqdm import tqdm

def train_ctc(epoch, dataloader, optimizer, model, loss_function, scheduler=None, *args, **kwargs):
    buffer = 0
    counter = 0

    model.train()
    for num, batch in tqdm(enumerate(dataloader), desc=f"Training classic approach - epoch {epoch}"):
        optimizer.zero_grad()

        softmaxed_output = torch.nn.functional.log_softmax(model(batch)['language_head_output'], dim=-1)

        ground_truth = batch['labels']
        print('outs', batch['output_lengths'])
        print('ins', batch['input_lengths'])


        loss = loss_function(softmaxed_output, ground_truth,
                             tuple([softmaxed_output.shape[0] for _ in range(softmaxed_output.shape[1])]),
                             tuple(batch['output_lengths']))
        print('loss', loss.item())
        if loss.item()!=loss.item():
            print(batch['transcriptions'])
            exit()
        loss.backward()
        optimizer.step()

        counter += 1
        buffer += loss.item()

    print(buffer / (num + 1))

