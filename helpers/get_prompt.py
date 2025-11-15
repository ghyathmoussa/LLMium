arabic_prompt = """استنادًا إلى النص العربي التالي، قم بإنشاء {} أسئلة ثاقبة مع إجاباتها المقابلة.
    يجب أن تكون الأسئلة والإجابات باللغة العربية.
    تأكد من أن الإجابات مستمدة مباشرة من النص المقدم.
    يجب أن يكون الناتج قائمة JSON تحتوي على كائنات، كل كائن بالصيغة التالية:

    النص:
    {}

    يجب أن يكون الناتج قائمة JSON تحتوي على كائنات، كل كائن بالصيغة التالية:
    [
      {{"question": "السؤال الأول هنا", "answer": "الإجابة الأولى هنا"}},
      {{"question": "السؤال الثاني هنا", "answer": "الإجابة الثانية هنا"}}
      // ... وهكذا
    ]
"""
#TODO add english and turkish prompts
english_prompt = """Based on the following English text, create {} insightful questions with their corresponding answers.
    The questions and answers should be in English.
    Ensure that the answers are derived directly from the provided text.

    Text:
    {}

    The output should be a JSON list containing objects, each in the following format:
    [
      {{"question": "First question here", "answer": "First answer here"}},
      {{"question": "Second question here", "answer": "Second answer here"}}
      // ... and so on
    ]
"""
turkish_prompt = """Aşağıdaki Türkçe metne dayanarak, {} içgörücü sorular ve karşılık gelen cevaplarını oluşturun.
    Sorular ve cevaplar Türkçe olmalıdır.
    Cevapların, sağlanan metinden doğrudan türetildiğinden emin olun.

    Metin:
    {}

    Çıktı, aşağıdaki formatta nesneler içeren bir JSON listesi olmalıdır:
    [
      {{"question": "İlk soru burada", "answer": "İlk cevap burada"}},
      {{"question": "İkinci soru burada", "answer": "İkinci cevap burada"}}
      // ... ve devamı
    ]
"""
def get_prompt(language, num_qa_pairs, text_content):
    """
    Returns the appropriate prompt based on the specified language and number of Q&A pairs.
    Args:
        language (str): The language for the prompt ('arabic', 'english', 'turkish').
        num_qa_pairs (int): The number of Q&A pairs to generate.
        text_content (str): The text content to base the questions on.
    Returns:
        str: The formatted prompt string.
    """
    if language == 'arabic':
        return arabic_prompt.format(num_qa_pairs, text_content)
    elif language == 'english':
        return english_prompt.format(num_qa_pairs, text_content)
    elif language == 'turkish':
        return turkish_prompt.format(num_qa_pairs, text_content)
    else:
        raise ValueError("Unsupported language. Please use 'arabic', 'english', or 'turkish'.")