# test_chatbot.py
from chatbot import MedicalRAG

def main():
    print("== Interactive Medical Chatbot ==")
    print("Type 'exit' to quit.\n")

    bot = MedicalRAG()
    session_id = "test_session"
    bot.start_session(session_id)

    while True:
        patient_input = input("You: ")
        if patient_input.lower().strip() in ["exit", "quit", "q"]:
            print("Chatbot: Goodbye! Take care.")
            break

        response = bot.ask(session_id, patient_input)
        print(f"Chatbot: {response}\n")

if __name__ == "__main__":
    main()
