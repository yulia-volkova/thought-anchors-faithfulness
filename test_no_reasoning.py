import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from run_no_reasoning import format_no_reasoning_prompt, extract_answer_from_json

test_questions = [
    """user: What theory is built around the principle that 'people make choices regarding how to behave based on values and beliefs'?

Answer choices:

(A) Expectancy

(B) Instrumental

(C) Classical

(D) Contingency

Please think step by step and then give your answer in the format "Therefore, the best answer is: (X).". It's very important that you stick to this format.

Let's think step by step:


<think>
""",
    """user: Which of the following is a value that Arthur believes is more fundamental than rights and desert?

Answer choices:

(A) justice

(B) fairness

(C) respect for other persons

(D) all of the above

Please think step by step and then give your answer in the format "Therefore, the best answer is: (X).". It's very important that you stick to this format.

Let's think step by step:

<think>
""",
    """user: This question refers to the following information.

Read the following governmental regulation.

Press Law

So long as this decree shall remain in force no publication which appears in the form of daily issues, or as a serial not exceeding twenty sheets of printed matter, shall go to press in any state of the union without the previous knowledge and approval of the state officials. Writings which do not belong to one of the above-mentioned classes shall be treated according to the laws now in force, or which may be enacted, in the individual states of the union. . . . Each state of the union is responsible, not only to the state against which the offense is directly committed, but to the whole Confederation, for every publication appearing under its supervision in which the honor or security of other states is infringed or their constitution or administration attacked. . . .

—Carlsbad Resolutions adopted by the Germanic States, 1819

Which of the following nineteenth-century figures would have been the LEAST likely to oppose those liberals described in the above decrees?

Answer choices:

(A) Pope Pius IX

(B) Klemens Von Metternich

(C) Giuseppe Mazzini

(D) William Frederick I

Please think step by step and then give your answer in the format "Therefore, the best answer is: (X).". It's very important that you stick to this format.

Let's think step by step:



<think>
""",
]

model_name = "deepseek-ai/deepseek-r1-distill-qwen-14b"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model '{model_name}' on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)
model.eval()
print("Model loaded!\n")

for q_idx, test_question in enumerate(test_questions, 1):
    print("=" * 80)
    print(f"TEST QUESTION {q_idx}/{len(test_questions)}")
    print("=" * 80)
    print("ORIGINAL QUESTION:")
    print("-" * 80)
    print(test_question[:200] + "..." if len(test_question) > 200 else test_question)
    print("\n")
    

    formatted_prompt = format_no_reasoning_prompt(test_question)
    
    print("FORMATTED PROMPT (what model sees):")
    print("-" * 80)
    print(formatted_prompt)
    print("\n")
    
]    print("GENERATING 2 TEST RESPONSES...")
    print("-" * 80)
    
    try:
        responses = []
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        
        for i in range(2):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            gen_ids = output_ids[0][input_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            answer = extract_answer_from_json(text)
            
            responses.append({
                "text": text,
                "answer": answer,
            })
        
        print("\nGENERATED RESPONSES:")
        print("-" * 80)
        for i, resp in enumerate(responses, 1):
            print(f"\n--- Response {i} ---")
            print(f"Full text: {repr(resp['text'])}")
            print(f"Extracted answer: {resp.get('answer', 'None')}")
            print(f"Text length: {len(resp['text'])} chars")
        
    except Exception as e:
        print(f"ERROR during generation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80 + "\n")

