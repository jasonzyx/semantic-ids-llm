SYSTEM_PROMPT = "You are a helpful AI assistant that understands and works with semantic IDs for product recommendations. Semantic IDs are hierarchical identifiers in the format <|sid_start|><|sid_105|><|sid_307|><|sid_705|><|sid_769|><|sid_end|> that encode product relationships and categories. /no_think"

REC_TEST_PROMPTS = [
    [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "User bought: <|sid_start|><|sid_240|><|sid_430|><|sid_558|><|sid_768|><|sid_end|>\n<|rec|>",
        },
    ],
    [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "User bought: <|sid_start|><|sid_240|><|sid_430|><|sid_558|><|sid_768|><|sid_end|>\n<|rec|>",
        },
    ],
    [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": 'Item ID: <|sid_start|><|sid_105|><|sid_307|><|sid_705|><|sid_769|><|sid_end|>\nRecommend other items in "Video Games":\n<|rec|>',
        },
    ],
]

TEST_PROMPTS = [
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "What is the title of item <|sid_start|><|sid_205|><|sid_402|><|sid_599|><|sid_768|><|sid_end|>?",
            },
        ],
        "description": "SID to title",
        "expected": "Grand Theft Auto: Liberty City Stories - PSP",
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "What is the description of product <|sid_start|><|sid_155|><|sid_402|><|sid_679|><|sid_768|><|sid_end|>?",
            },
        ],
        "description": "SID to description",
        "expected": "Enhance your PlayStation 2 gaming experience with this special bundle. It includes the PlayStation 2 console and one controller, plus a bonus Dual Shock 2 controller for multiplayer action. Get started with three exciting games: Madden NFL 2001, SSX: Snowboard Supercross, and Ridge Racer V. The PlayStation 2 features a 128-bit Emotion Engine processor for fast gameplay and stunning graphics. Madden NFL 2001 offers realistic football action with detailed player and field graphics. SSX: Snowboard Supercross delivers fast-paced winter sports with unique environments. Ridge Racer V showcases the console's power with visually impressive vehicles and dazzling reflective surfaces. This bundle also includes an official PS2 vertical stand, allowing you to display your console stylishly.",
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": 'Product title "Disney Infinity 3.0 Tomorrowland Power Disc Pack (PS4, Xbox One, PS3, Xbox 360, Wii)" maps to ID:',
            },
        ],
        "description": "Title to SID",
        "expected": "<|sid_start|><|sid_26|><|sid_345|><|sid_764|><|sid_768|><|sid_end|>",
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Customers shopping in Video Games > Legacy Systems > Nintendo Systems > Nintendo 64 > Accessories also shop:",
            },
        ],
        "description": "Category tree transition",
        "expected": "Video Games > Legacy Systems > Nintendo Systems > Super Nintendo > Accessories",
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "User bought: <|sid_start|><|sid_25|><|sid_256|><|sid_536|><|sid_768|><|sid_end|>, <|sid_start|><|sid_197|><|sid_388|><|sid_676|><|sid_768|><|sid_end|>, <|sid_start|><|sid_199|><|sid_404|><|sid_555|><|sid_768|><|sid_end|>\n<|rec|>",
            },
        ],
        "description": "Sequence last 3",
        "expected": "<|sid_start|><|sid_230|><|sid_451|><|sid_534|><|sid_768|><|sid_end|>",
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "User bought: <|sid_start|><|sid_240|><|sid_430|><|sid_558|><|sid_768|><|sid_end|>\n<|rec|>",
            },
        ],
        "description": "Sequence last 1",
        "expected": "<|sid_start|><|sid_56|><|sid_262|><|sid_564|><|sid_768|><|sid_end|>",
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": 'Item ID: <|sid_start|><|sid_105|><|sid_307|><|sid_705|><|sid_769|><|sid_end|>\nRecommend other items in "Video Games":\n<|rec|>',
            },
        ],
        "description": "Recommend based on category",
        "expected": "<|sid_start|><|sid_139|><|sid_382|><|sid_741|><|sid_769|><|sid_end|>, <|sid_start|><|sid_149|><|sid_355|><|sid_586|><|sid_768|><|sid_end|>, <|sid_start|><|sid_173|><|sid_324|><|sid_764|><|sid_768|><|sid_end|>",
    },
    # New test cases for explanations, bundle naming, and follow-up conversations
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "I have <|sid_start|><|sid_148|><|sid_393|><|sid_545|><|sid_768|><|sid_end|>. What would you recommend and why?",
            },
        ],
        "description": "Recommendation with explanation",
        "expected": "<|sid_start|><|sid_105|><|sid_307|><|sid_705|><|sid_769|><|sid_end|>\n\nIf you liked the Nyko Charge Base",
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Suggest a catchy name for a bundle containing: <|sid_start|><|sid_197|><|sid_427|><|sid_597|><|sid_768|><|sid_end|>, <|sid_start|><|sid_75|><|sid_259|><|sid_656|><|sid_768|><|sid_end|>, <|sid_start|><|sid_75|><|sid_495|><|sid_752|><|sid_768|><|sid_end|>",
            },
        ],
        "description": "Bundle name",
        "expected": "Nintendo RPG Adventure Pack",
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Why would these three products work well together: <|sid_start|><|sid_51|><|sid_387|><|sid_612|><|sid_768|><|sid_end|>, <|sid_start|><|sid_155|><|sid_489|><|sid_713|><|sid_768|><|sid_end|>, <|sid_start|><|sid_113|><|sid_403|><|sid_760|><|sid_768|><|sid_end|>?",
            },
        ],
        "description": "Bundle description",
        "expected": "Experience epic adventures and intense combat",
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "User bought: <|sid_start|><|sid_148|><|sid_393|><|sid_545|><|sid_768|><|sid_end|>\n<|rec|>",
            },
            {
                "role": "assistant",
                "content": "<|sid_start|><|sid_105|><|sid_307|><|sid_705|><|sid_769|><|sid_end|>",
            },
            {
                "role": "user",
                "content": "Why is this a good recommendation?",
            },
        ],
        "description": "Two-turn: Recommendation + Explanation",
        "expected": "Both are essential accessories for enhancing the PlayStation",
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Recommend products with Activision like <|sid_start|><|sid_144|><|sid_434|><|sid_605|><|sid_768|><|sid_end|>:\n<|rec|>",
            },
            {
                "role": "assistant",
                "content": "<|sid_start|><|sid_64|><|sid_449|><|sid_584|><|sid_768|><|sid_end|>, <|sid_start|><|sid_177|><|sid_260|><|sid_753|><|sid_768|><|sid_end|>, <|sid_start|><|sid_219|><|sid_259|><|sid_625|><|sid_768|><|sid_end|>",
            },
            {
                "role": "user",
                "content": "Name this recommended bundle:",
            },
        ],
        "description": "Two-turn: Brand rec + Bundle name",
        "expected": "Call of Duty",
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "User bought: <|sid_start|><|sid_105|><|sid_341|><|sid_652|><|sid_768|><|sid_end|>\n<|rec|>",
            },
            {
                "role": "assistant",
                "content": "<|sid_start|><|sid_144|><|sid_434|><|sid_605|><|sid_768|><|sid_end|>",
            },
            {
                "role": "user",
                "content": "I prefer First-Person Shooter games, any recommendations?",
            },
        ],
        "description": "Two-turn: Seq last + Genre refinement",
        "expected": "<|sid_start|><|sid_96|><|sid_283|><|sid_543|><|sid_768|><|sid_end|>",
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "User bought: <|sid_start|><|sid_144|><|sid_499|><|sid_568|><|sid_768|><|sid_end|>\n<|rec|>",
            },
            {
                "role": "assistant",
                "content": "<|sid_start|><|sid_219|><|sid_299|><|sid_575|><|sid_768|><|sid_end|>",
            },
            {
                "role": "user",
                "content": "Can you suggest similar games on PS2?",
            },
        ],
        "description": "Two-turn: Seq last + Platform refinement",
        "expected": "<|sid_start|><|sid_197|><|sid_487|><|sid_529|><|sid_768|><|sid_end|>",
    },
]
