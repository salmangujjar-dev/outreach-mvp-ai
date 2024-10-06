# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import AzureOpenAI
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

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
    print(request)
    try:
        model = AzureOpenAI()

        prompt1 = ChatPromptTemplate.from_template(
            """
            Imagine you are the best copywriter in the world, crafting irresistible email subject lines.

            Guidelines:
            - Personalize the subject line using "{job_title}" and "{jobCompanyName}" for a personal touch.
            - Keep it casual, conversational, and under 7 words. Mimic the writing style: "{writing_style}".
            - Examples: "{first_name}, take a look!" or "Exciting update for {jobCompanyName}".
            - Avoid emojis, cliches, or overused phrases.
            - Each subject line should build upon previous ones to maintain flow and continuity.

            Output Format:
            - Return the result as JSON with the key "subject".
            """
        )

        prompt2 = ChatPromptTemplate.from_template(
            """
            Imagine you are the best copywriter in the world, crafting an engaging email opening.

            Guidelines:
            - Personalize the first paragraph using "{summary}" (education summary), "{twitter_activity}" (recent Twitter activity), or combine "{job_title}" and "{jobCompanyName}" for relevance.
            - Write in a casual, conversational style, under 30 words, mimicking the style: "{writing_style}".
            - Example: "Hi {first_name}, I saw your work at {jobCompanyName} and had to reach out."
            - Avoid emojis, cliches, or generic language.

            Output Format:
            - Return the result as JSON with the key "content".
            """
        )

        chain1 = prompt1 | model | StrOutputParser()
        chain2 = {"email": chain1} | prompt2 | model | StrOutputParser()

        email_content = await chain2.invoke({
            "summary": request.lead.education.summary,
            "twitter_activity": request.lead.twitterUrl,
            "job_title": request.lead.jobTitle,
            "jobCompanyName": request.lead.jobCompanyName,
            "first_name": request.lead.firstName,
            "writing_style": request.persona.icpQuestions.customerSupport,
            "usp": request.persona.icpQuestions.usp,
            "industry": request.lead.industry,
            "full_name": request.lead.fullName
        })

        return {"email": email_content}
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate email")