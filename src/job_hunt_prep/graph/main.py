"""
main.py

Here we can do a trial run for our agent.

J. A. Moreno
2026
"""

from .agent import Agent
from .llm_setup import LLM, Retriever

def main():
    print("Testing our agent\n")

    user_query = """
This is the job post:https://recruiting2.ultipro.com/INF1019IRINC/JobBoard/17a8d008-9efe-4e51-8460-47ee205d5229/OpportunityDetail?opportunityId=70528dc0-415e-4c8d-970f-46cf2a253bd0
They are asking: We are looking for people who are personally and/or
professionally passionate about AI. Please briefly explain how you have
put it to work for you in either or both areas of your life.
    """.strip()

    #Initialize agent

    agent = Agent(llm=LLM().llm, retriever=Retriever().retriever)
    agent.draw_mermaid_png()
    
    for part in agent.run_agent(user_query):
        if part["type"] == "messages":
            # MessagesStreamPart — (message_chunk, metadata) from LLM calls
            msg, metadata = part["data"]
            print(msg.content, end="", flush=True)
    
    user_query = """
They are asking: What are your three main motivators for this position?

Additional instructions: Keep your answer below 500 characters.
    """.strip()

    for part in agent.run_agent(user_query):
        if part["type"] == "messages":
            # MessagesStreamPart — (message_chunk, metadata) from LLM calls
            msg, metadata = part["data"]
            print(msg.content, end="", flush=True)
    
    user_query = """
What was a challenge you found during your past experience and how did you solved it?

User insructions: Keep your answer below 250 characters, avoid technical jargon and use references
no older than 5 years.
    """.strip()

    for part in agent.run_agent(user_query):
        if part["type"] == "messages":
            # MessagesStreamPart — (message_chunk, metadata) from LLM calls
            msg, metadata = part["data"]
            print(msg.content, end="", flush=True)

    user_query = """
Here's the job post URL: https://mx.indeed.com/viewjob?jk=b88cd0e9cc3d52c0&from=shareddesktop_copy

Within the URL, extract both the job post and questions.
        """.strip()
    for part in agent.run_agent(user_query):
        if part["type"] == "messages":
            # MessagesStreamPart — (message_chunk, metadata) from LLM calls
            msg, metadata = part["data"]
            print(msg.content, end="", flush=True)
    

    user_query = """
    Here are the job post text and the questions:
    Descripción completa del empleo

Senior AI Developer (LLMs &amp; Python)

Sobre el Rol

Buscamos un Senior AI Developer con un fuerte ADN de ingeniería de software para liderar la implementación de soluciones de Inteligencia Artificial Generativa. Tu misión será diseñar y desarrollar servicios escalables que impacten áreas críticas como Desarrollo (Dev), Calidad (QA) y Operaciones (Ops), reactivando proyectos estratégicos como eStratis.

Si te apasiona la orquestación de modelos (LLMs), el diseño de arquitecturas asíncronas y quieres llevar la IA de un prototipo a un entorno productivo real, ¡esta posición es para ti!

Responsabilidades Clave

    Desarrollo de Soluciones IA: Construcción de servicios, APIs y componentes robustos basados en modelos de lenguaje (LLMs) para uso transversal en la organización.
    Arquitectura y Escalabilidad: Definir y evolucionar arquitecturas técnicas desacopladas (Microservicios, Event-Driven) para soluciones de IA en producción.
    Impulso a Proyectos Estratégicos: Liderar la continuidad de eStratis (Dev AI) y habilitar capacidades de IA para los equipos de QA y Ops.
    Colaboración Técnica: Trabajar de la mano con Arquitectura, DevOps y stakeholders de negocio para alinear la tecnología con los objetivos estratégicos.
    Innovación: Evaluar y adoptar modelos (GPT, Claude, Llama) y herramientas de vanguardia para resolver problemas reales.

¿Qué buscamos en ti? (Habilidades Indispensables)

    Formación: Licenciatura en Ingeniería en Sistemas, Computación, Matemáticas, Ciencia de Datos o afín.
    Experiencia Backend: 5+ años desarrollando software con fuerte enfoque en Python (obligatorio).
    Expertiz en IA/ML: Experiencia sólida implementando soluciones en entornos productivos.
    Dominio de LLMs: Experiencia práctica con modelos como GPT, Claude, Gemini, Llama o Mistral y su integración vía APIs.
    Orquestación de Modelos: Uso de frameworks como LangChain o LlamaIndex.
    Infraestructura Cloud: Experiencia con plataformas de hosting y orquestación como AWS Bedrock.
    Arquitectura de Software: Diseño de sistemas escalables, microservicios y arquitecturas dirigidas por eventos (Event-Driven).
    Cultura Ops: Conocimientos en MLOps / LLMOps (versionado de modelos, monitoreo y control de calidad) y pipelines de CI/CD.

Puntos Extra (Deseables)

    Experiencia con QA basado en IA o automatización de testing.
    Experiencia en sectores industriales, financieros o entornos enterprise.
    Habilidad evaluando calidad de prompts y outputs de modelos.
    Conocimientos de seguridad, privacidad y compliance en IA.

Condiciones de Trabajo

    Esquema: Híbrido (Mayormente Home Office con visitas ocasionales a sitio).
    Ubicación: Insurgentes Sur 1079, Col. Noche Buena, Benito Juárez, CDMX.
    Asignación: Tiempo Indefinido.

Tipo de puesto: Tiempo completo

Sueldo: A partir de $75,000.00 al mes

Beneficios:

    Programa de referidos
    Seguro de gastos médicos mayores
    Seguro de vida

Pregunta(s) de postulación:

    ¿Cuántos años de experiencia profesional tienes desarrollando software con Python (Backend)?
    ¿Has implementado soluciones con LLMs (GPT, Llama, Claude) en entornos de producción real? (Sí/No)
    ¿Cuántos años de experiencia tienes utilizando servicios de AWS (específicamente Bedrock, SageMaker o EC2)?
    ¿Con qué framework de orquestación de LLMs tienes más experiencia profesional?

Lugar de trabajo: remoto híbrido en 03720, Nochebuena, CDMX

        """.strip()
    for part in agent.run_agent(user_query):
        if part["type"] == "messages":
            # MessagesStreamPart — (message_chunk, metadata) from LLM calls
            msg, metadata = part["data"]
            print(msg.content, end="", flush=True)

if __name__ == "__main__":
    main()
