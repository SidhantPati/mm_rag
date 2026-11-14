from conversation import response_generation

user_prompt = "What is this company about?"
response, context = response_generation(user_prompt,[])

print(f"\nUser: {user_prompt}\n")
print(f"\nAI: {response}\n")
print(f"\nCONTEXT: {context}\n")