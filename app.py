import gradio as gr
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")

tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")


this_description = '''
Using facebook/m2m100_1.2B pre-trained model. Language code:
English(en)
Turkish(tr)
German(de)
'''


def m2m_translate(Input_Text, from_lang, to_lang):
    tokenizer.src_lang = from_lang

    encoded_from_lang = tokenizer(Input_Text, return_tensors="pt")

    generated_tokens = model.generate(
        **encoded_from_lang, forced_bos_token_id=tokenizer.get_lang_id(to_lang))

    res = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return res[0]


iface = gr.Interface(
    fn=m2m_translate,

    title="M2M100 Translation",
    description=this_description,

    inputs=[
        gr.inputs.Textbox(lines=5, placeholder="Enter text"),

        gr.inputs.Radio(
            choices=[
                'de',
                'en',
                'tr',
            ],
            default='vi',
            label='From language'),

        gr.inputs.Radio(
            choices=[
                'de',
                'en',
                'tr',
            ],
            default='en',
            label='To language'),
    ],
    outputs="text")

iface.launch()
