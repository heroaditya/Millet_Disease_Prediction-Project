def get_chatbot_response(user_input):
    responses = {
        "hello": "Hello! How can I assist you with millet farming?",
        "disease": "I can help with disease identification. Upload an image for analysis.",
        "fertilizer": "For millets, organic compost and NPK fertilizers work best.",
        "What should I do if my crop has fungus?": "Use organic fungicides like neem oil.",
        "How to prevent millet diseases?": "Ensure proper irrigation and rotate crops regularly.",
        "How to identify a treatable disease?": "Early-stage fungal infections are often treatable.",
    }
    return responses.get(user_input, "I'm not sure. Please consult an expert.")

