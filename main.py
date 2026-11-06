"""Streamlit user interface for Smart Medical Note Assistant."""

import streamlit as st
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, BadRequestError, InternalServerError, RateLimitError
from pydantic import ValidationError
from medical_service import SUPPORTED_ROLES, build_prompt, generate_clinical_note, process_notes
from validation import validate_clinical_note

st.set_page_config(page_title="Smart Medical Note Assistant", page_icon="🩺", layout="wide")
st.title("🩺 Smart Medical Note Assistant")
st.caption("Turn unstructured practitioner notes into structured SOAP documentation.")
st.info("For demonstration and documentation support only. Use fictional data. A qualified clinician must review every generated note.")

with st.form("clinical_note_form"):
    practitioner_role = st.selectbox("Practitioner role", options=sorted(SUPPORTED_ROLES), index=0)
    practitioner_notes = st.text_area("Practitioner notes", placeholder="- Patient reports fever for 2 days\n- Headache\n- Temperature: 101°F\n- No vomiting", height=220, max_chars=10_000, help="Maximum 10,000 characters. Do not enter identifiable patient data.")
    submitted = st.form_submit_button("Generate SOAP Note", type="primary", use_container_width=True)

if submitted:
    try:
        cleaned_notes = process_notes(practitioner_notes)
        prompt = build_prompt(notes=cleaned_notes, user_role=practitioner_role, note_type="SOAP")
        with st.spinner("Generating and validating the SOAP note..."):
            response_text = generate_clinical_note(prompt)
            soap_note = validate_clinical_note(response_text)
        st.success("SOAP note generated. Please review it before use.")
        for heading, content in [("Subjective", soap_note.subjective), ("Objective", soap_note.objective), ("Assessment", soap_note.assessment), ("Plan", soap_note.plan)]:
            st.subheader(heading)
            st.write(content)
        st.download_button("Download SOAP note (JSON)", data=soap_note.model_dump_json(indent=2), file_name="soap_note.json", mime="application/json")
    except ValueError as error:
        st.warning(str(error))
    except ValidationError:
        st.error("The generated response did not match the required SOAP structure. Please try again.")
    except AuthenticationError:
        st.error("Gemini authentication failed. Check your API key.")
    except BadRequestError:
        st.error("Gemini rejected the request. Check the input and model configuration.")
    except RateLimitError:
        st.error("Gemini rate limit or quota reached. Please try again later.")
    except (APITimeoutError, APIConnectionError):
        st.error("Could not reach Gemini in time. Check your connection and retry.")
    except InternalServerError:
        st.error("Gemini is temporarily unavailable. Please try again later.")
    except APIStatusError as error:
        st.error(f"Gemini request failed (HTTP {error.status_code}). Check your model configuration.")
    except Exception:
        st.error("An unexpected error occurred while generating the note.")
