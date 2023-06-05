from transformers import pipeline, LlamaForCausalLM, AutoTokenizer, GenerationConfig, Conversation
from peft import LoraConfig, TaskType
from peft.peft_model import PeftModel
import torch
import os

class CastOutputToFloat(torch.nn.Sequential):

    def forward(self, x):
        return super().forward(x)


checkpoint = 'output/checkpoint-220'
base_model_path = '/workspace/llama-7b'


def load_model(checkpoint, base_model_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)

    model = LlamaForCausalLM.from_pretrained(base_model_path)
    peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
                             target_modules=["up_proj"])
    output_embedding_layer = getattr(model, 'lm_head')
    setattr(model, 'lm_head', CastOutputToFloat(output_embedding_layer))
    model = PeftModel(model, peft_config)

    state_dict = torch.load(os.path.join(checkpoint, 'pytorch_model.bin'))
    model.load_state_dict(state_dict)
    model.eval()
    model.half()
    return model, tokenizer

model, tokenizer = load_model(checkpoint, base_model_path)

# prompt = "Somatic hypermutation allows the immune system to"
# [{'generated_text': 'Somatic hypermutation allows the immune system to recognize and attack cancer cells.\nThe research'}]

# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
# print(prompt)
# complete_sentence = generator(prompt)

generation_config = GenerationConfig(
    max_length=1024,
    num_beams=4,
    do_sample=True,
    early_stopping=True,
    temperature=0.5,
)
model.generation_config = generation_config
# You could then use the named generation config file to parameterize generation
# inputs = tokenizer("translate English to French: Configuration files are easy to use!", return_tensors="pt")

# outputs = model.generate(**inputs, generation_config=generation_config)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

chatbot = pipeline(model=model, tokenizer=tokenizer)
conversation = Conversation("")
x = ""
while x.lower() != 'end':
    x = input('User: ')
    conversation.add_user_input(x)
    conversation = chatbot(conversation)
    print('Bot:', conversation.generated_responses[-1])
