# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Education(BaseModel):
    school: str
    linkedinUrl: str
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    degreeName: Optional[str] = None
    raw: List[str] = []
    summary: str


class ICPQuestions(BaseModel):
    usp: Optional[str] = None
    industry: Optional[str] = None
    customerSupport: Optional[str] = None


class Persona(BaseModel):
    name: str
    icpQuestions: ICPQuestions


class LeadDTO(BaseModel):
    _id: str
    uniqueIdentifier: str
    firstName: str
    lastName: str
    fullName: str
    linkedinUrl: str
    linkedinUsername: str
    linkedinId: str
    twitterUrl: str
    twitterUsername: str
    workEmail: str
    industry: str
    jobTitle: str
    jobCompanyName: str
    jobCompanyWebsite: str
    jobCompanyIndustry: str
    jobCompany12moEmployeeGrowthRate: float
    jobCompanyTotalFundingRaised: float
    jobCompanyInferredRevenue: Optional[str] = None
    jobCompanyEmployeeCount: int
    jobLastChanged: str
    jobLastVerified: str
    jobStartDate: str
    jobCompanySize: Optional[str] = None
    jobCompanyFounded: int
    jobCompanyLocationRegion: str
    locationName: str
    locationCountry: str
    skills: List[str]
    education: Education
    gender: str
    companyEmployees: str
    dataProvider: str
    __v: int

class EmailRequest(BaseModel):
    lead: LeadDTO
    persona: Persona
    probability: str

@app.post("/v1/email")
async def generate_email(request: EmailRequest):
    try:
        model = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

        input_data = {
            "education_summary": request.lead.education.summary,
            "job_title": request.lead.jobTitle,
            "job_company_name": request.lead.jobCompanyName,
            "first_name": request.lead.firstName,
            "writing_style": request.persona.icpQuestions.customerSupport,
            "usp": request.persona.icpQuestions.usp,
            "customer_support": request.persona.icpQuestions.customerSupport,
            "industry": request.lead.industry,
            "full_name": request.lead.fullName,
        }

        prompt1 = ChatPromptTemplate.from_template(
            """
            Imagine you are the best copywriter in the world, crafting irresistible email subject lines.

            Guidelines:
            - Personalize the subject line using "{job_title}" and "{job_company_name}" and "{industry}" for a personal touch.
            - Keep it casual, conversational, and under 7 words.
            - Examples: "{first_name}, take a look!" or "Exciting update for {job_company_name}".
            - Avoid emojis, cliches, or overused phrases.
            - Do not add any extra explanation to the response. Just give the output.

            Output Format:
            - Provide the final email format as JSON with the key "subject".
            """
        )

        prompt2 = ChatPromptTemplate.from_template(
            """
            Imagine you are the best copywriter in the world, crafting an engaging email opening.

            Guidelines:
            - Personalize the first paragraph using the "{subject}", "{education_summary}", or combine "{job_title}" and "{job_company_name}" for relevance.
            - Also, consider the targeting persona that includes unique selling point "{usp}" and "{customer_support}.
            - Remember to always include the first name i.e. {first_name}.
            - Write in a casual, conversational style, Professional style, and under 30 words.
            - Example: "Hi {first_name}, I saw your work at {job_company_name} and had to reach out."
            - Avoid emojis, cliches, or generic language.
            - Do not add any extra explanation to the response. Just give the output.

            Output Format:
            - Provide the final email format as JSON with the key "content".
            """
        )

        chain1 = prompt1 | model | SimpleJsonOutputParser()
        subject_response = chain1.invoke(input_data)
        subject = subject_response.get("subject", "")

        chain2 = prompt2 | model | SimpleJsonOutputParser()
        body_response = chain2.invoke({**input_data, "subject": subject})
        body = body_response.get("content", "")

        response = {"subject": subject, "body": body}

        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate email")