# @Time   : 2022/9/30
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

from crslab.data.dataset.tokenizer.base import BaseCrsTokenize

from nltk import word_tokenize
import nltk


class nltk_tokenize(BaseCrsTokenize):

    def __init__(self, path=None) -> None:
        super().__init__(path)

    def tokenize(self, text):
        nltk.download('punkt')
        return word_tokenize(text)