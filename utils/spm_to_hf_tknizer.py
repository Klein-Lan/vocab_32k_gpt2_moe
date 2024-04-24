#import tokenizers
#ython sentencepiece_extractor.py  --provider sentencepiece --model ../configs/tokenizer_models/32k_vocab_guyu_pajama_pj.model --vocab-output-path ../configs/tokenizer_models/vocab.json --merges-output-path ../configs/tokenizer_models/merges.txt
#tok=tokenizers.SentencePieceBPETokenizer.from_spm("../configs/tokenizer_models/32k_vocab_guyu_pajama_pj.model")
#from transformers import AutoTokenizer
#sp = AutoTokenizer.from_pretrained('../configs/tokenizer_models')
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='../configs/tokenizer_models/32k_vocab_guyu_pajama_pj.model')
print(sp.encode("a good man 你是好人"))
print(sp.unk_token)
#vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
#print(vocabs)
#tok.save_pretrained("hf_format_tokenizer")

from models.tokenization_guyu import GuyuTokenizer

tokenizer = GuyuTokenizer(vocab_file='../configs/tokenizer_models/32k_vocab_guyu_pajama_pj.model',legacy=False)
print(tokenizer.encode("a good man 你是好人"))

'''
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast

# SentencePiece模型文件路径
sentencepiece_model_path = "../configs/tokenizer_models/32k_vocab_guyu_pajama_pj.model"

# 加载SentencePiece模型
sentencepiece_tokenizer = SentencePieceBPETokenizer(sentencepiece_model_path,unk_token= "<unk>")

print(sentencepiece_tokenizer.encode("a good man 你是好人"))
# 保存为Hugging Face Transformers的tokenizer
sentencepiece_tokenizer.save_model("tmp2")

# 加载Hugging Face Transformers的tokenizer
huggingface_tokenizer = PreTrainedTokenizerFast.from_pretrained("tmp2")

# 使用Hugging Face Transformers的tokenizer进行编码
encoded_input = huggingface_tokenizer("a good man 你是好人")
print(encoded_input)
'''
