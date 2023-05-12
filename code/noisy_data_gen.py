import os
import pandas as pd
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import warnings
warnings.filterwarnings("ignore")


def prepare_input(aug_mode, filepath, mode=''):
    df = pd.read_csv(filepath, sep=' ', header=None, skip_blank_lines=False)
    df.columns = ['in_token', 'tag']

    list_sent_df = []
    ind = df['in_token'].isnull().to_numpy().nonzero()[0]
    ind = np.concatenate(([0], ind, [len(df)]))

    if mode == 'BI_O':
        return df, ind

    symbol_pos = {}
    sent_df_len = pd.DataFrame(columns=['sent_ind', 'sent_len'])
    for sent_ind, (start, end) in enumerate(zip(ind, ind[1:])):
        if start == 0:
            sent_df = df[start: end]
        else:
            sent_df = df[start: end].iloc[1:]

        if mode == 'BI':
            sent_df = sent_df[(sent_df['tag'] == 'B') | (sent_df['tag'] == 'I')]
        else:
            # word_swap: identify and delete rows with only symbols, then record deleted indices within each sent_df
            sent_df_len = sent_df_len.append({'sent_ind': sent_ind, 'sent_len': len(sent_df)}, ignore_index=True)
            sent_df_len['cum_sent_len'] = sent_df_len['sent_len'].cumsum()

            flag = (sent_df['in_token'].str.isalnum()) & (sent_df['tag'] == 'O')      # symbol + O tag
            tmp_df = sent_df[~flag].reset_index()
            if sent_ind != 0:
                tmp_df['index'] -= sent_df_len.loc[sent_ind-1, 'cum_sent_len'] + sent_ind
            symbol_pos[str(sent_ind)] = tmp_df
            sent_df = sent_df[flag]

        list_sent_df.append(sent_df)
    return list_sent_df, symbol_pos


def prepare_output(aug_mode, list_df, list_sent, symbol_pos, filepath, mode='', beta=False):
    num_sent = len(list_df)
    df_list = []
    for ind in range(num_sent):
        out_list_words = word_tokenize(list_sent[ind])
        df = pd.concat([list_df[ind].reset_index(drop=True), pd.DataFrame(out_list_words)], axis=1, ignore_index=True)

        if len(df) == 0:
            df = pd.DataFrame({'in_token': '', 'out_token': '', 'tag': ''}, index=[0])
        else:
            df.columns = ['in_token', 'tag', 'out_token']

        df = pd.concat([df, pd.DataFrame({'in_token': '', 'out_token': '', 'tag': ''}, index=[0])], ignore_index=True)

        if len(symbol_pos) != 0:    # aug_mode == 'word_swap'
            symbol_df = symbol_pos[str(ind)]
            symbol_df['out_token'] = symbol_df['in_token']
            for idx, row in symbol_df.iterrows():
                pos = symbol_df['index'].loc[idx]
                line = symbol_df[['in_token', 'tag', 'out_token']].loc[idx]
                df = pd.concat([df.iloc[:pos], line.to_frame().T, df.iloc[pos:]]).reset_index(drop=True)

            if aug_mode == 'word_swap':
                df = df.iloc[:-1].sample(frac=1).reset_index(drop=True)
                df = pd.concat([df, pd.DataFrame({'in_token': '', 'out_token': '', 'tag': ''}, index=[0])], ignore_index=True)

        df = df[['in_token', 'out_token', 'tag']]

        if beta:
            df.to_csv(os.path.splitext(filepath)[0] + '.csv', header=False, index=False, mode='a')

        out_df = df[['out_token', 'tag']]
        if len(mode) == 0:
            out_df.to_csv(filepath, header=False, index=False, sep=' ', mode='a')
        else:
            df_list.append(df)
    return df_list


# https://github.com/makcedward/nlpaug/blob/23800cbb9632c7fc8c4a88d46f9c4ecf68a96299/example/textual_augmenter.ipynb
def text_augment(text, aug_mode, aug_char_p=0.3, aug_word_p=0.3):
    if aug_mode == 'char_keyboard':
        aug = nac.KeyboardAug(aug_char_p=aug_char_p, aug_word_p=aug_word_p,
                              stopwords=list(stopwords.words('english')),
                              include_special_char=False, include_numeric=False)
    elif aug_mode == 'char_swap':
        aug = nac.RandomCharAug(action='swap', swap_mode='random',
                                aug_char_p=aug_char_p, aug_word_p=aug_word_p,
                                stopwords=list(stopwords.words('english')), spec_char='')
    elif aug_mode == 'word_spelling':
        aug = naw.SpellingAug(aug_p=aug_word_p, stopwords=list(stopwords.words('english')))
    elif aug_mode == 'word_swap':
        aug = naw.RandomWordAug(action='swap', aug_p=aug_word_p)
    elif aug_mode == 'word_synonym':    # to be updated: avoid generating synonym based on embedding similarity
        aug = naw.SynonymAug(aug_src='wordnet', aug_p=aug_word_p, stopwords=list(stopwords.words('english')))
    res = aug.augment(text)
    return res


def detokenize(in_list_sent_df):
    return [TreebankWordDetokenizer().detokenize(sent_df['in_token'].to_list()) for sent_df in in_list_sent_df]


def run(aug_mode, in_filepath, out_filepath, aug_char_p, aug_word_p):
    in_list_sent_df, symbol_pos = prepare_input(aug_mode, in_filepath, mode='')
    out_list_sent = text_augment(detokenize(in_list_sent_df), aug_mode, aug_char_p, aug_word_p)
    prepare_output(aug_mode, in_list_sent_df, out_list_sent, symbol_pos, out_filepath, beta=True)


def run_BI_O(aug_mode, in_filepath, out_filepath, BI_aug_char_p, BI_aug_word_p, O_aug_char_p, O_aug_word_p, beta):
    print('BI_O mode...')
    BI_in_list_sent_df, BI_symbol_pos = prepare_input(aug_mode, in_filepath, mode='BI')
    BI_df_list = prepare_output(aug_mode, BI_in_list_sent_df,
                                text_augment(detokenize(BI_in_list_sent_df), aug_mode, BI_aug_char_p, BI_aug_word_p),
                                BI_symbol_pos, out_filepath, mode='BI')

    O_in_list_sent_df, O_symbol_pos = prepare_input(aug_mode, in_filepath, mode='O')
    O_df_list = prepare_output(aug_mode, O_in_list_sent_df,
                               text_augment(detokenize(O_in_list_sent_df), aug_mode, O_aug_char_p, O_aug_word_p),
                               O_symbol_pos, out_filepath, mode='BI')

    df, ind = prepare_input(aug_mode, in_filepath, mode='BI_O')
    for sent_ind, (start, end) in enumerate(zip(ind, ind[1:])):
        if start == 0:
            sent_df = df[start: end]
        else:
            sent_df = df[start: end].iloc[1:]

        BI_df, O_df = BI_df_list[sent_ind][:-1], O_df_list[sent_ind][:-1]
        output = pd.merge(BI_df, O_df, how='outer', on=['in_token', 'out_token'], suffixes=('_BI', '_O'))
        output['tag'] = output['tag_O'].fillna(output['tag_BI'])
        output = output.drop(columns=['tag_O', 'tag_BI'])
        output = pd.concat([output, pd.DataFrame({'in_token': '', 'out_token': '', 'tag': ''}, index=[0])], ignore_index=True)

        output = pd.merge(sent_df, output.drop_duplicates(), on=['in_token', 'tag'], how='inner')
        output = pd.concat([output, pd.DataFrame({'in_token': '', 'out_token': '', 'tag': ''}, index=[0])], ignore_index=True)
        output = output[['in_token', 'out_token', 'tag']]

        if beta:
            output.to_csv(os.path.splitext(out_filepath)[0] + '.csv', header=False, index=False, mode='a')

        output[['out_token', 'tag']].to_csv(out_filepath, header=False, index=False, sep=' ', mode='a')


def wrapper(aug_mode_list, in_filepath):
    aug_char_p, aug_word_p = 0.3, 0.3

    for aug_mode in aug_mode_list:
        print('aug_mode: ', aug_mode)
        run(aug_mode, in_filepath, in_filepath.split('.')[0] + f'-output-{aug_mode}-{aug_char_p}-{aug_word_p}.txt',
            aug_char_p, aug_word_p)


def wrapper_BIO(aug_mode_list, in_filepath):
    BI_aug_char_p, BI_aug_word_p = 0.5, 0.5
    O_aug_char_p, O_aug_word_p = 0.2, 0.2

    for aug_mode in aug_mode_list:
        if aug_mode == 'word_swap':
            pass
        print('aug_mode: ', aug_mode)
        run_BI_O(aug_mode, in_filepath,
                 in_filepath.split('.')[0] + f'-output-{aug_mode}-BI-{BI_aug_char_p}_{BI_aug_char_p}-O-{O_aug_char_p}_{O_aug_word_p}.txt',
                 BI_aug_char_p, BI_aug_word_p, O_aug_char_p, O_aug_word_p, beta=True)


if __name__ == "__main__":
    in_filepath = 'test.txt'
    wrapper(['char_keyboard', 'char_swap', 'word_swap'], in_filepath)
    wrapper_BIO(['char_keyboard', 'char_swap'], in_filepath)

