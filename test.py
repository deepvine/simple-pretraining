from transformers import pipeline, BertTokenizer



class Arg:
    tokenizer_name = "./test-mlm-hancom/vocab.txt"

args = Arg()


tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name, do_lower_case=True)


fill_mask = pipeline(
    "fill-mask",
    model="./test-mlm-hancom",
    tokenizer= tokenizer
)


result = fill_mask("나는 어제 [MASK]")
print(result)

while True:
    inputs = input(">")
    print(fill_mask(inputs))