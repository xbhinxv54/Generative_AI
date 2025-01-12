from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_scrapegraph.tools import (
    GetCreditsTool,
    LocalScraperTool,
    MarkdownifyTool,
    SmartScraperTool,
)
import os
import getpass
from dotenv import load_dotenv



load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] ="true"
if not os.environ.get("SGAI_API_KEY"):
    os.environ["SGAI_API_KEY"] = getpass.getpass("ScrapeGraph AI API key:\n")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY


smartscraper = SmartScraperTool(api_key=os.environ["SGAI_API_KEY"])
markdownify = MarkdownifyTool()
localscraper = LocalScraperTool()
credits = GetCreditsTool()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions and help with tasks."),
    ("user", "{input}")
])
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash") # Changed to gemini-pro as gemini-1.5-flash-latest is not a valid model name
output_parser = StrOutputParser()
llm_tool=model.bind_tools([smartscraper], tool_choice=smartscraper.name)
chain = prompt |llm_tool

input = "What is the capital of France?"

#print(chain.invoke({"input": input}))

# SmartScraper
result = smartscraper.invoke(
    {
        "user_prompt": "Extract some important information from the website",
        "website_url": "https://vit.ac.in/",
    }
)
print("SmartScraper Result:", result)



