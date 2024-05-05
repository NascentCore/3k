from langchain_openai import OpenAI
from langchain.chains import ConversationChain

# first initialize the large language model
llm = OpenAI(
	temperature=0,
	openai_api_key="OPENAI_API_KEY",
	model_name="text-davinci-003"
)

# now initialize the conversation chain
conversation = ConversationChain(llm=llm)

print(conversation.prompt.template)