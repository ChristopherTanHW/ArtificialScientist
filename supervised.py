from __future__ import unicode_literals, print_function, division
import pickle

from io import open
import unicodedata
import re

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import random

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

infile = open('pickle_dataset','rb')
print('loading data...')
dataset = pickle.load(infile)
infile.close()

class LinearEncoder(nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
teacher_forcing_ratio = 1
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, 
          decoder_optimizer, criterion, vocab_size, max_length=10):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    #print('target_length: ', target_length)

    loss = 0
    
    latent = encoder(input_tensor)

    #decoder_input = F.one_hot(torch.arange(vocab_size))[0]
    decoder_input = torch.tensor([[0]], device=device)
    #print('first decoder_input: ', decoder_input)

    decoder_hidden = latent.view(1,1,-1)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            #print('recur decoder_input: ', decoder_input)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            #print('decoder output: ', decoder_output)
            #print('target tensor di: ', target_tensor[di].view(1))
            loss += criterion(decoder_output, target_tensor[di].view(1).long())
            decoder_input = target_tensor[di].view(1).long() # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break


    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def pad_rollouts(rollout, max_len):
    rollout_len = rollout.size()[0]
    if rollout_len > max_len - 1:
        print('rollout too long')
        return None
    remaining = max_len - rollout_len
    padding = torch.zeros((remaining, 1, 1, 6, 6, 1, 8))
    padded_rollout = torch.cat((rollout, padding), 0)
    return padded_rollout

#pads all relevant statement embeddings to some max len and returns their flattened tensor
def preprocess_relstat_emb(relstat_emb, max_len = 15):
    l = torch.zeros([1])
    torch_relstat = torch.tensor(relstat_emb)
    l = torch.cat((l,torch_relstat))
    if l.size(0) > max_len:
        print('relstat too long')
        return None
    while l.size()[0] < max_len:
        l = torch.cat((l, torch.ones([1])))
        l = l.flatten()
    return l

def flatten_all(dataset, max_len_rollout, max_len_relstat):
    flat_dataset = []
    for i in range(len(dataset)):
        entry = []
        flat_task = torch.flatten(dataset[i]['task_emb'])
        padded_rollout = pad_rollouts(dataset[i]['rollout'], max_len_rollout)
        flat_rollout = padded_rollout.flatten()
        processed_relstat = preprocess_relstat_emb(dataset[i]['relstat_emb'], max_len_relstat)
        entry.append(torch.cat((flat_task, flat_rollout)))
        entry.append(processed_relstat)
        flat_dataset.append(entry)
    return flat_dataset

def trainIters(dataset, encoder, decoder, n_iters, output_size, print_every=100, plot_every=10, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = dataset
    criterion = nn.NLLLoss()
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, output_size)
        #print('loss: ', loss)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            
    showPlot(plot_losses)
    return plot_losses

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, input_tensor, vocab, max_length=10):
    with torch.no_grad():
        latent = encoder(input_tensor)

        decoder_input = torch.tensor([[0]], device=device)  # SOS

        decoder_hidden = latent.view(1,1,-1)

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == 1:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(vocab[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words
    
def evaluateRandomly(encoder, decoder, pairs, vocab, n=10):
    for i in range(n):
        pair = pairs[i]
        target_statements = idx_to_sent(pair[1], vocab)
        print('target statements: ', target_statements)
        output_words = evaluate(encoder, decoder, pair[0], vocab)
        print('output_words: ', output_words)
        #output_sentence = ' '.join(output_words)
        #print('<', output_sentence)
        #print('')

vocab = ['pad', 'eos', '', 'wall', 'door', 'orcishdagger', 'dagger', 'silverdagger', 'athame', 
         'elvendagger', 'wormtooth', 'knife', 'stiletto', 'scalpel', 'crysknife', 'axe', 'battleaxe', 
         'pickaxe', 'dwarvishmattock', 'orcishshortsword', 'shortsword', 'dwarvishshortsword', 
         'elvenshortsword', 'broadsword', 'runesword', 'elvenbroadsword', 'longsword', 'katana', 
         'twohandedsword', 'tsurugi', 'scimitar', 'silversaber', 'club', 'aklys', 'mace', 'morningstar', 
         'flail', 'grapplinghook', 'warhammer', 'quarterstaff', 'partisan', 'fauchard', 'glaive', 'becdecorbin', 
         'spetum', 'lucernhammer', 'guisarme', 'ranseur', 'voulge', 'billguisarme', 'bardiche', 'halberd', 
         'orcishspear', 'spear', 'silverspear', 'elvenspear', 'dwarvishspear', 'javelin', 'trident', 'lance', 
         'orcishbow', 'orcisharrow', 'bow', 'arrow', 'elvenbow', 'elvenarrow', 'yumi', 'ya', 'silverarrow', 
         'sling', 'flintstone', 'crossbow', 'crossbowbolt', 'dart', 'shuriken', 'boomerang', 'bullwhip', 
         'rubberhose', 'unicornhorn', 'unarmed', 'basedagger', 'baseknive', 'baseaxe', 'basepickaxe', 
         'baseshortsword', 'basebroadsword', 'baselongsword', 'basetwohandedsword', 'basescimitar', 
         'basesaber', 'baseclub', 'basemace', 'basemorningstar', 'baseflail', 'basehammer', 'basequarterstave', 
         'basepolearm', 'basespear', 'basetrident', 'baselance', 'basebow', 'basesling', 'basecrossbow', 'basedart', 
         'baseshuriken', 'baseboomerang', 'basewhip', 'baseunicornhorn', 'hawaiianshirt', 'tshirt', 'leatherjacket', 
         'leatherarmor', 'orcishringmail', 'studdedleatherarmor', 'ringmail', 'scalemail', 'orcishchainmail', 
         'chainmail', 'elvenmithrilcoat', 'splintmail', 'bandedmail', 'dwarvishmithrilcoat', 'bronzeplatemail', 
         'platemail', 'crystalplatemail', 'dragonscales', 'mummywrapping', 'orcishcloak', 'dwarvishcloak', 
         'leathercloak', 'oilskincloak', 'fedora', 'dentedpot', 'elvenleatherhelm', 'helmet', 'orcishhelm', 
         'dwarvishironhelm', 'leathergloves', 'smallshield', 'orcishshield', 'urukhaishield', 'elvenshield', 
         'dwarvishroundshield', 'largeshield', 'lowboots', 'highboots', 'ironshoes', 'baseshirt', 'basesuit', 
         'basedragonsuit', 'basecloak', 'basehelm', 'baseglove', 'baseshield', 'baseboot', 'amulet', 'ring', 
         'weapon', 'armour', 'accessory', 'agent', 'player', 'queuedagent', 'monster', 'stationarymonster', 
         'dragon', 'hostilemonster', 'randommonster', 'structure', 'unobservable', 'empty', 'baseitem', 
         'basemonster', 'move', 'around', 'welcome', 'to', 'rtfm', '.', 'cold', 'fire', 'poison', 'lightning', 
         'you', 'are', 'beat', ',', '{', '}', 'wolf', 'jaguar', 'panther', 'goblin', 'bat', 'imp', 'shaman', 
         'ghost', 'zombie', 'grandmasters', 'blessed', 'shimmering', 'gleaming', 'fanatical', 'mysterious', 
         'soldiers', 'arcane', 'star', 'alliance', 'order', 'of', 'the', 'forest', 'rebel', 'enclave', 'sword', 
         'polearm', 'cutlass', 'modifiers', 'weapons', 'element', 'monsters', 'items', 'effective', 'against', 
         'defeated', 'by', 'should', 'use', 'get', 'slay', 'useful', 'for', 'good', 'not', 'weak', 'slain', 
         'group', 'belong', 'contains', 'on', 'team', 'consists', 'same', '-', 'they', 'has', 'following', 
         'members', ':', 'is', 'made', 'up', 'make', 'defeat', 'must', 'be', 'needs', 'beaten', 'evil', 'fight', 
         'in', 'from']

print("preparing data...")
flat_dataset = flatten_all(dataset, 30, 15)
random.shuffle(flat_dataset)

input_size = flat_dataset[0][0].size()[0]
print('input size: ', input_size)

output_size = 262
latent_size = 256
encoder = LinearEncoder(input_size, latent_size).to(device)
decoder = DecoderRNN(latent_size, output_size).to(device)

n_iters = 100000

print("training...")
losses = trainIters(flat_dataset, encoder, decoder, n_iters, output_size, print_every=100, plot_every=100, learning_rate=0.001)