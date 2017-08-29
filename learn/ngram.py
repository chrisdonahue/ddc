from collections import Counter
import random

class NgramSequence:
    def __init__(self, chart_notes):
        self.sequence = [sym for _, _, _, sym in chart_notes]

    def get_ngrams(self, k, pre=True, post=True):
        prepend = []
        if pre:
            prepend = ['<pre{}>'.format(i) for i in reversed(range(k - 1))]

        append = []
        if post:
            append = ['<post>']

        sequence = prepend + self.sequence + append
        for i in xrange(len(sequence) - (k - 1)):
            yield tuple(sequence[i:i + k])

class NgramLanguageModel:
    def __init__(self, k, ngram_counts):
        self.k = k
        self.ngram_counts = ngram_counts
        self.ngram_total = sum(self.ngram_counts.values())

        self.history_counts = Counter()
        for ngram, count in ngram_counts.items():
            self.history_counts[ngram[:-1]] += count

        self.vocab = set()
        for ngram, _ in ngram_counts.items():
            for w in ngram:
                self.vocab.add(w)

    def mle(self, ngram):
        ngram_count = self.ngram_counts[ngram]
        history = ngram[:-1]
        history_count = self.history_counts[ngram[:-1]]
        #print ngram, ngram_count, history_count
        return float(ngram_count) / history_count

    def laplace(self, ngram, smooth=1):
        history_count = self.history_counts.get(ngram[:-1], 0)
        card_v = len(self.vocab) + 1
        numerator = 1 + self.ngram_counts.get(ngram, 0)
        denominator = card_v + history_count

        return float(numerator) / denominator

    def generate(self, history, strategy='argmax'):
        if strategy == 'argmax':
            highest_prob = 0
            best_chars = [None]
            for v in self.vocab:
                v_prob = self.laplace(history + (v,))
                if v_prob > highest_prob:
                    highest_prob = v_prob
                    best_chars = [v]
                elif v_prob == highest_prob:
                    best_chars.append(v)
            return random.choice(best_chars)
        else:
            raise NotImplementedError()

if __name__ == '__main__':
    import argparse
    from collections import Counter
    import cPickle as pickle
    import json
    import math
    import os
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_fp', type=str, help='')
    parser.add_argument('model_fp', type=str, help='')
    parser.add_argument('--k', type=int, help='')
    parser.add_argument('--diff', type=str, help='')
    parser.add_argument('--task', type=str, choices=['train', 'eval'], help='')

    parser.set_defaults(
        k=1,
        task='train',
        diff='')

    args = parser.parse_args()

    with open(args.dataset_fp, 'r') as f:
        json_fps = f.read().split()

    if args.task == 'train':
        ncharts = 0
        ngram_counts = Counter()

        for json_fp in json_fps:
            with open(json_fp, 'r') as f:
                song_meta = json.loads(f.read())

            for chart_meta in song_meta['charts']:
                if args.diff and args.diff != chart_meta['difficulty_coarse']:
                    continue
                chart_sequence = NgramSequence(chart_meta['notes'])
                for ngram in chart_sequence.get_ngrams(args.k):
                    ngram_counts[ngram] += 1
                ncharts += 1

        model = NgramLanguageModel(args.k, ngram_counts)

        ptotmle = 0.0
        for ngram, count in ngram_counts.items():
            ptotmle += model.mle(ngram)

        with open(args.model_fp, 'wb') as f:
            pickle.dump(model, f)

    elif args.task == 'eval':
        with open(args.model_fp, 'rb') as f:
            model = pickle.load(f)

        chart_entropies = []
        chart_accuracies = []
        for json_fp in json_fps:
            with open(json_fp, 'r') as f:
                song_meta = json.loads(f.read())
            #print song_meta['title']

            for chart_meta in song_meta['charts']:
                if args.diff and args.diff != chart_meta['difficulty_coarse']:
                    continue
                chart_sequence = NgramSequence(chart_meta['notes'])
                chart_log_prob = 0.0
                chart_n = 0
                hits = 0
                for ngram in chart_sequence.get_ngrams(args.k, pre=True, post=False):
                    generated = model.generate(ngram[:-1])
                    actual = ngram[-1]
                    if generated == actual:
                        hits += 1
                    ngram_prob = model.laplace(ngram)
                    #print '{}: {}'.format(ngram, ngram_prob)
                    chart_log_prob += np.log(ngram_prob)
                    chart_n += 1
                cross_entropy = (-1.0 / chart_n) * chart_log_prob
                chart_entropies.append(cross_entropy)
                chart_accuracies.append(float(hits) / chart_n)

        chart_entropies = np.array(chart_entropies)
        chart_perplexities = np.exp(chart_entropies)
        #print chart_entropies
        #print chart_perplexities
        #print chart_entropies
        #print chart_perplexities
        #print chart_accuracies
        #print 'Cross-entropy (nats): {}, std {}'.format(np.mean(chart_entropies), np.std(chart_entropies))
        #print 'Perplexity: {}, std {}'.format(np.mean(chart_perplexities), np.std(chart_perplexities))
        #print 'Accuracy: {}, std {}'.format(np.mean(chart_accuracies), np.std(chart_accuracies))
        #print '-' * 30 + 'COPY PASTA' + '-' * 30
        eval_funcs = [np.mean, np.std, np.min, np.max]
        eval_results = [[f(x) for f in eval_funcs] for x in [chart_entropies, chart_perplexities, chart_accuracies]]
        copy_pasta = []
        for l in eval_results:
            copy_pasta.append(','.join([str(x) for x in l]))
        print ','.join(copy_pasta)
    else:
        raise NotImplementedError()
