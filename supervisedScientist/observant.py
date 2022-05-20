from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re

import torch
import typing
import numpy as np
import random
import itertools
import pandas as pd
import string
import matplotlib.pyplot as plt
import pickle

import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import math

import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# importing pickled dataset of rollouts
infile = open('supervisedScientist/pickle_dataset','rb')
dataset = pickle.load(infile)
infile.close()

EOS_token = 1

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, 
          decoder_optimizer, criterion, vocab_size, teacher_forcing_ratio, max_length=15):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    #print('target_length: ', target_length)

    loss = 0

    decoder_input = torch.tensor([[0]], device=device)

    decoder_hidden = encoder(input_tensor).view(1,1,-1)
    
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

            loss += criterion(decoder_output, target_tensor[di].view(1).long())
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

def trainIters(dataset, encoder, decoder, n_iters, output_size, teacher_forcing_low, print_every=100, plot_every=10, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = dataset
    criterion = nn.NLLLoss()
    
    
    for iter in range(1, n_iters + 1):
        teacher_forcing_ratio = 1 - (iter / n_iters) * teacher_forcing_low
        
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, output_size, teacher_forcing_ratio)
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

print('preparing dataset...')
flat_dataset = flatten_all(dataset, 30, 15)
flat_train = flat_dataset[:99000]
flat_test = flat_dataset[1000:]
random.shuffle(flat_train)

input_size = flat_dataset[0][0].size()[0]
print('input size: ', input_size)

from linearEncoder import LinearEncoder
from decoder import DecoderRNN

output_size = 262
latent_size = 200

encoder = LinearEncoder(input_size, latent_size).to(device)
decoder = DecoderRNN(latent_size, output_size).to(device)

n_iters = 10000

print('training...')

losses = trainIters(flat_train, encoder, decoder, n_iters, output_size, 0.5, print_every=100, plot_every=100, learning_rate=0.001)

plt.plot(losses)

targets, preds = evaluateRandomly(encoder, decoder, flat_train, vocab, n=10)

for i in range(len(targets)):
    print('target: ', targets[i])
    print('preds: ', preds[i])

def clean_pred(pred):
    newls = []
    for l in pred:
        if l == 'pad' or l == '<EOS>':
            continue
        newls.append(l)
    return newls

def clean_eval(targets, preds):
    cleant = []
    for t in targets:
        clean_tar = clean_target(t)
        cleant.append(clean_tar)
    
    cleanp = []
    for p in preds:
        clean_pre = clean_pred(p)
        cleanp.append(clean_pre)
    return cleant, cleanp

def clean_target(target):
    ls = target.split()
    newls = []
    for l in ls:
        if l == 'pad' or l == 'eos':
            continue
        newls.append(l)
    return newls

def evaluate(encoder, decoder, input_tensor, vocab, max_length=15):
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
    target = []
    predicted = []
    for i in range(n):
        pair = pairs[i]
        target_statements = idx_to_sent(pair[1], vocab)
        target.append(target_statements)
        output_words = evaluate(encoder, decoder, pair[0], vocab)
        predicted.append(output_words)
        #output_sentence = ' '.join(output_words)
        #print('<', output_sentence)
        #print('')
    return target, predicted

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

def idx_to_sent(idxs, vocab):
    sent = ''
    idxs = idxs.int()
    for i in idxs:
        sent += vocab[i] +' '
    return sent