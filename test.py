from transformers import pipeline, BertTokenizer, DistilBertTokenizer



class Arg:
    tokenizer_name = "./tokenizer/wordpiece/vocab.txt"
    model = "./model/han-bert"

args = Arg()


tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name, do_lower_case=True)
# tokenizer = DistilBertTokenizer.from_pretrained(args.tokenizer_name, do_lower_case=True)


fill_mask = pipeline(
    "fill-mask",
    model=args.model,
    tokenizer=tokenizer
)


# sample
result = fill_mask("나는 어제 [MASK]")
print(result)

while True:
    inputs = input(">")
    print(fill_mask(inputs))