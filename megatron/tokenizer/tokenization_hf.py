import collections
# from transformers.models.bert.tokenization_bert import BasicTokenizer, WordpieceTokenizer, load_vocab
from transformers.models.bert.tokenization_bert import load_vocab
from .bert_tokenization import BasicTokenizer, WordpieceTokenizer


class MyBertTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        # never_split = None
        # tokenize_chinese_chars = True
        # strip_accents = None
        self.unk_token = "[UNK]"
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        # self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        output = []
        for token in tokens:
            output.append(self.vocab.get(token, self.vocab.get(self.unk_token)))
        return output

    def convert_ids_to_tokens(self, indexs):
        output = []
        for index in indexs:
            output.append(self.ids_to_tokens.get(index, self.unk_token))
        return output

    @staticmethod
    def convert_tokens_to_string(tokens, clean_up_tokenization_spaces=False):
        """ Converts a sequence of tokens (string) in a single string. """

        def clean_up_tokenization(out_string):
            """ Clean up a list of simple English tokenization artifacts
            like spaces before punctuations and abreviated forms.
            """
            out_string = (
                out_string.replace(" .", ".")
                    .replace(" ?", "?")
                    .replace(" !", "!")
                    .replace(" ,", ",")
                    .replace(" ' ", "'")
                    .replace(" n't", "n't")
                    .replace(" 'm", "'m")
                    .replace(" 's", "'s")
                    .replace(" 've", "'ve")
                    .replace(" 're", "'re")
            )
            return out_string

        text = ' '.join(tokens).replace(' ##', '').strip()
        if clean_up_tokenization_spaces:
            clean_text = clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def vocab_size(self):
        return len(self.vocab)
