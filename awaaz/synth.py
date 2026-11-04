"""Synthetic multilingual grievance generator.

Real labelled civic-grievance data for Telugu/Hindi/Urdu doesn't exist publicly, so we
compose complaints from hand-written building blocks:

    frame( locality , issue clause , duration sentence , extra sentence )

* every issue clause carries a base urgency (0=low .. 3=critical)
* duration / vulnerability sentences can bump urgency (never above "high" unless the
  issue itself is a hazard) — so the model has to *read context*, not just keywords
* safety rules are applied to the final text so labels and the rule layer agree

Phrase-level hold-out: for each (department, language) cell a subset of issue clauses is
used ONLY in the test split, so test metrics measure generalisation to unseen phrasings.
"""
from __future__ import annotations

import random

from . import config, rules
from .text import script_of

LANGS = ["en", "te", "hi", "ur", "tenglish", "hinglish"]

# ---------------------------------------------------------------------------- frames
# {loc} is already in locative form (e.g. "in Kukatpally", "కూకట్‌పల్లిలో", "Kukatpally mein").
FRAMES = {
    "en": ["{issue} {loc}. {dur} {extra}",
           "Sir, {issue} {loc}. {extra}",
           "Complaint {loc}: {issue}. {dur}",
           "Hello, I stay {loc}. {issue}. {dur} {extra}",
           "{issue} near our colony {loc}. {extra}",
           "Respected officer, {issue} {loc}. {dur} {extra}"],
    "te": ["{loc} {issue}. {dur} {extra}",
           "సార్, మా {loc} {issue}. {extra}",
           "నమస్తే, నేను {loc} ఉంటాను. {issue}. {dur} {extra}",
           "{loc} మా కాలనీలో {issue}. {extra}",
           "ఫిర్యాదు: {loc} {issue}. {dur}"],
    "hi": ["{loc} {issue}। {dur} {extra}",
           "सर, {loc} {issue}। {extra}",
           "नमस्ते, मैं {loc} रहता हूँ। {issue}। {dur} {extra}",
           "शिकायत: {loc} {issue}। {dur}",
           "{loc} हमारी कॉलोनी में {issue}। {extra}"],
    "ur": ["{loc} {issue}۔ {dur} {extra}",
           "جناب، {loc} {issue}۔ {extra}",
           "السلام علیکم، میں {loc} رہتا ہوں۔ {issue}۔ {dur} {extra}",
           "شکایت: {loc} {issue}۔ {dur}"],
    "tenglish": ["{loc} {issue}. {dur} {extra}",
                 "Sir, maa {loc} {issue}. {extra}",
                 "Namaskaram, nenu {loc} untanu. {issue}. {dur} {extra}",
                 "Complaint: {loc} {issue}. {dur}",
                 "{loc} maa colony lo {issue}. {extra}"],
    "hinglish": ["{loc} {issue}. {dur} {extra}",
                 "Sir, {loc} {issue}. {extra}",
                 "Namaste, main {loc} rehta hoon. {issue}. {dur} {extra}",
                 "Complaint: {loc} {issue}. {dur}",
                 "{loc} hamari colony mein {issue}. {extra}"],
}

LOC_FORMS = {
    "en": "in {name}", "te": "{name}లో", "hi": "{name} में", "ur": "{name} میں",
    "tenglish": "{name} lo", "hinglish": "{name} mein",
}
NO_LOC = {"en": "in our area", "te": "మా ప్రాంతంలో", "hi": "हमारे इलाके में", "ur": "ہمارے علاقے میں",
          "tenglish": "maa area lo", "hinglish": "hamare area mein"}

# --------------------------------------------------------------------- duration (text, bump)
DURATIONS = {
    "en": [("", 0), ("This started yesterday.", 0), ("It has been like this for 3 days.", 0),
           ("It has been more than a week.", 1), ("This problem has continued for almost a month.", 1),
           ("For two weeks nothing has been done.", 1)],
    "te": [("", 0), ("నిన్నటి నుండి ఇలా ఉంది.", 0), ("మూడు రోజులుగా ఇలాగే ఉంది.", 0),
           ("వారం రోజులు దాటింది.", 1), ("దాదాపు నెల రోజులుగా ఇదే సమస్య.", 1)],
    "hi": [("", 0), ("यह कल से हो रहा है।", 0), ("तीन दिन से ऐसा ही है।", 0),
           ("एक हफ्ते से ज़्यादा हो गया है।", 1), ("लगभग एक महीने से यही समस्या है।", 1)],
    "ur": [("", 0), ("یہ کل سے ہو رہا ہے۔", 0), ("تین دن سے ایسا ہی ہے۔", 0),
           ("ایک ہفتے سے زیادہ ہو گیا ہے۔", 1), ("تقریباً ایک مہینے سے یہی مسئلہ ہے۔", 1)],
    "tenglish": [("", 0), ("Ninnati nundi ila undi.", 0), ("Moodu rojuluga ilaage undi.", 0),
                 ("Vaaram rojulu daatindi.", 1), ("Daadapu nela rojuluga ide samasya.", 1)],
    "hinglish": [("", 0), ("Ye kal se ho raha hai.", 0), ("Teen din se aisa hi hai.", 0),
                 ("Ek hafte se zyada ho gaya hai.", 1), ("Lagbhag ek mahine se yahi problem hai.", 1)],
}

# ------------------------------------------------------------- extra sentence (text, bump)
EXTRAS = {
    "en": [("", 0), ("Please take action.", 0), ("Kindly resolve this at the earliest.", 0),
           ("Nobody is responding to our calls.", 0), ("Many elderly people and small children live here.", 1),
           ("School children pass through here every day.", 1), ("This is my third complaint about this.", 1)],
    "te": [("", 0), ("దయచేసి చర్యలు తీసుకోండి.", 0), ("వీలైనంత త్వరగా పరిష్కరించండి.", 0),
           ("ఎవరూ మా ఫోన్లకు స్పందించడం లేదు.", 0), ("ఇక్కడ చాలా మంది వృద్ధులు, చిన్న పిల్లలు ఉంటారు.", 1),
           ("స్కూల్ పిల్లలు రోజూ ఇక్కడి నుండే వెళ్తారు.", 1), ("ఇది నా మూడవ ఫిర్యాదు.", 1)],
    "hi": [("", 0), ("कृपया कार्रवाई करें।", 0), ("जल्द से जल्द ठीक करवाइए।", 0),
           ("कोई हमारी बात नहीं सुन रहा।", 0), ("यहाँ बहुत सारे बुज़ुर्ग और छोटे बच्चे रहते हैं।", 1),
           ("स्कूल के बच्चे रोज़ यहाँ से गुज़रते हैं।", 1), ("यह मेरी तीसरी शिकायत है।", 1)],
    "ur": [("", 0), ("براہ کرم کارروائی کریں۔", 0), ("جلد از جلد ٹھیک کروائیں۔", 0),
           ("کوئی ہماری بات نہیں سن رہا۔", 0), ("یہاں بہت سے بزرگ اور چھوٹے بچے رہتے ہیں۔", 1),
           ("یہ میری تیسری شکایت ہے۔", 1)],
    "tenglish": [("", 0), ("Dayachesi action teesukondi.", 0), ("Veelainantha twaraga fix cheyyandi.", 0),
                 ("Evaru maa phones ki respond avvatledu.", 0),
                 ("Ikkada chala mandi peddavallu, chinna pillalu untaru.", 1),
                 ("School pillalu roju ikkadi nunde velthaaru.", 1), ("Idi naa moodo complaint.", 1)],
    "hinglish": [("", 0), ("Please action lijiye.", 0), ("Jaldi se jaldi theek karwaiye.", 0),
                 ("Koi hamari baat nahi sun raha.", 0), ("Yahan bahut saare buzurg aur chhote bachche rehte hain.", 1),
                 ("School ke bachche roz yahan se guzarte hain.", 1), ("Ye meri teesri complaint hai.", 1)],
}

# ----------------------------------------------------------- issue clauses (text, base urgency)
ISSUES: dict[str, dict[str, list[tuple[str, int]]]] = {
    "roads": {
        "en": [("the road is full of big potholes", 1), ("there is a huge pothole in the middle of the main road", 2),
               ("the road has caved in and a deep pit has formed", 3), ("road digging work was left half done", 1),
               ("the speed breaker is broken and vehicles are skidding", 2), ("the footpath tiles are broken", 0),
               ("the newly laid road is already damaged", 0), ("rain water is stagnating on the damaged road", 1)],
        "hi": [("सड़क पर बहुत बड़े गड्ढे हैं", 1), ("मेन रोड के बीच में एक बहुत बड़ा गड्ढा है", 2),
               ("सड़क धँस गई है और गहरा गड्ढा बन गया है", 3), ("सड़क की खुदाई का काम आधा छोड़ दिया गया है", 1),
               ("स्पीड ब्रेकर टूटा हुआ है और गाड़ियाँ फिसल रही हैं", 2), ("फुटपाथ की टाइलें टूटी हुई हैं", 0),
               ("नई बनी सड़क अभी से खराब हो गई है", 0)],
        "te": [("రోడ్డు మీద పెద్ద పెద్ద గుంతలు ఉన్నాయి", 1), ("మెయిన్ రోడ్డు మధ్యలో చాలా పెద్ద గుంత ఉంది", 2),
               ("రోడ్డు కుంగిపోయి లోతైన గొయ్యి ఏర్పడింది", 3), ("రోడ్డు తవ్వకం పని సగంలో వదిలేశారు", 1),
               ("స్పీడ్ బ్రేకర్ విరిగిపోయింది, వాహనాలు జారిపోతున్నాయి", 2), ("ఫుట్‌పాత్ టైల్స్ పగిలిపోయాయి", 0),
               ("కొత్తగా వేసిన రోడ్డు అప్పుడే పాడైపోయింది", 0)],
        "ur": [("سڑک پر بہت بڑے گڑھے ہیں", 1), ("مین روڈ کے بیچ میں ایک بہت بڑا گڑھا ہے", 2),
               ("سڑک دھنس گئی ہے اور گہرا گڑھا بن گیا ہے", 3), ("سڑک کی کھدائی کا کام آدھا چھوڑ دیا گیا ہے", 1),
               ("فٹ پاتھ کی ٹائلیں ٹوٹی ہوئی ہیں", 0)],
        "hinglish": [("road pe bahut bade gaddhe hain", 1), ("main road ke beech mein bahut bada gaddha hai", 2),
                     ("road dhans gayi hai aur gehra gaddha ban gaya hai", 3),
                     ("road ki khudai ka kaam aadha chhod diya hai", 1),
                     ("speed breaker toota hua hai, gaadiyan fisal rahi hain", 2),
                     ("footpath ki tiles tooti hui hain", 0)],
        "tenglish": [("road meeda pedda pedda gunthalu unnayi", 1), ("main road madhyalo chala pedda guntha undi", 2),
                     ("road kungipoyi lothaina goyyi erpadindi", 3), ("road thavvakam pani sagamlo vadilesaru", 1),
                     ("speed breaker virigipoyindi, vehicles jaaripothunnayi", 2), ("footpath tiles pagilipoyayi", 0)],
    },
    "water": {
        "en": [("there has been no water supply", 2), ("drinking water is coming very dirty and smelly", 2),
               ("the water pipeline has burst and water is being wasted", 2),
               ("water pressure is very low, it does not reach the tank", 1),
               ("water comes only for 10 minutes at odd hours", 1), ("the water meter is not working", 0),
               ("the water tanker we booked never came", 1), ("water is leaking near the valve", 1)],
        "hi": [("पानी की सप्लाई बिल्कुल नहीं आ रही है", 2), ("पीने का पानी बहुत गंदा और बदबूदार आ रहा है", 2),
               ("पानी की पाइपलाइन फट गई है और पानी बर्बाद हो रहा है", 2),
               ("पानी का प्रेशर बहुत कम है, टंकी तक नहीं पहुँचता", 1), ("पानी सिर्फ दस मिनट के लिए आता है", 1),
               ("पानी का मीटर काम नहीं कर रहा है", 0), ("बुक किया हुआ पानी का टैंकर नहीं आया", 1)],
        "te": [("నీటి సరఫరా అసలు రావడం లేదు", 2), ("తాగునీరు చాలా మురికిగా, దుర్వాసనతో వస్తోంది", 2),
               ("నీటి పైప్‌లైన్ పగిలి నీళ్లు వృథా అవుతున్నాయి", 2),
               ("నీటి ప్రెషర్ చాలా తక్కువగా ఉంది, ట్యాంక్‌కి ఎక్కడం లేదు", 1),
               ("నీళ్లు కేవలం పది నిమిషాలే వస్తున్నాయి", 1), ("వాటర్ మీటర్ పనిచేయడం లేదు", 0),
               ("బుక్ చేసిన వాటర్ ట్యాంకర్ రాలేదు", 1)],
        "ur": [("پانی کی سپلائی بالکل نہیں آ رہی ہے", 2), ("پینے کا پانی بہت گندا اور بدبودار آ رہا ہے", 2),
               ("پانی کی پائپ لائن پھٹ گئی ہے اور پانی ضائع ہو رہا ہے", 2), ("پانی کا پریشر بہت کم ہے", 1),
               ("پانی کا میٹر کام نہیں کر رہا ہے", 0)],
        "hinglish": [("paani ki supply bilkul nahi aa rahi hai", 2),
                     ("peene ka paani bahut ganda aur badboodar aa raha hai", 2),
                     ("paani ki pipeline phat gayi hai, paani waste ho raha hai", 2),
                     ("paani ka pressure bahut kam hai, tank tak nahi chadhta", 1),
                     ("paani sirf das minute ke liye aata hai", 1), ("water meter kaam nahi kar raha", 0)],
        "tenglish": [("water supply asalu raavatledu", 2), ("manchineellu chala murikiga, vaasana tho vasthunnayi", 2),
                     ("water pipeline pagili neellu waste avthunnayi", 2),
                     ("water pressure chala thakkuva undi, tank ki ekkatledu", 1),
                     ("neellu kevalam padi nimishale vasthunnayi", 1), ("water meter pani cheyyatledu", 0)],
    },
    "sewerage": {
        "en": [("the drainage is overflowing onto the road", 2), ("sewage water is entering our houses", 2),
               ("the manhole is open and uncovered", 3), ("the drain is completely blocked with plastic", 1),
               ("there is a terrible stink from the open nala", 1), ("the drainage pipe is leaking near the gate", 1),
               ("sewage is mixing with the drinking water line", 3),
               ("the storm water drain needs cleaning before monsoon", 0)],
        "hi": [("नाली का पानी सड़क पर बह रहा है", 2), ("सीवेज का पानी हमारे घरों में घुस रहा है", 2),
               ("मैनहोल खुला पड़ा है, ढक्कन नहीं है", 3), ("नाली पूरी तरह प्लास्टिक से जाम है", 1),
               ("खुले नाले से बहुत बदबू आ रही है", 1), ("सीवेज पीने के पानी की लाइन में मिल रहा है", 3),
               ("बारिश से पहले नाले की सफाई करनी है", 0)],
        "te": [("డ్రైనేజీ పొంగి రోడ్డు మీదకు వస్తోంది", 2), ("మురుగు నీరు మా ఇళ్లలోకి వస్తోంది", 2),
               ("మ్యాన్‌హోల్ మూత లేదు, తెరిచి ఉంది", 3), ("డ్రైనేజీ పూర్తిగా ప్లాస్టిక్‌తో బ్లాక్ అయింది", 1),
               ("నాలా నుండి భరించలేని దుర్వాసన వస్తోంది", 1), ("మురుగు నీరు తాగునీటి పైప్‌లో కలుస్తోంది", 3),
               ("వర్షాకాలానికి ముందు నాలా శుభ్రం చేయాలి", 0)],
        "ur": [("نالی کا پانی سڑک پر بہہ رہا ہے", 2), ("سیوریج کا پانی ہمارے گھروں میں گھس رہا ہے", 2),
               ("مین ہول کھلا پڑا ہے، ڈھکن نہیں ہے", 3), ("نالی پلاسٹک سے بالکل بند ہے", 1),
               ("کھلے نالے سے بہت بدبو آ رہی ہے", 1)],
        "hinglish": [("naali ka paani road pe beh raha hai", 2), ("sewage ka paani hamare gharon mein ghus raha hai", 2),
                     ("manhole khula pada hai, dhakkan nahi hai", 3), ("naali poori tarah plastic se jam hai", 1),
                     ("khule naale se bahut badboo aa rahi hai", 1),
                     ("sewage peene ke paani ki line mein mil raha hai", 3)],
        "tenglish": [("drainage pongi road meedaki vasthondi", 2), ("muruguneeru maa illaloki vasthondi", 2),
                     ("manhole moota ledu, therichi undi", 3), ("drainage motham plastic tho block ayindi", 1),
                     ("nala nundi bharinchaleni vaasana vasthondi", 1),
                     ("drainage neellu manchineelli pipe lo kalusthunnayi", 3)],
    },
    "electricity": {
        "en": [("there has been no power", 2), ("a live electric wire has fallen on the road", 3),
               ("the transformer is sparking and making loud noise", 3), ("there are frequent power cuts every day", 1),
               ("voltage is fluctuating and appliances are getting damaged", 1),
               ("the electricity bill is wrongly calculated", 0), ("the electric pole is leaning dangerously", 2),
               ("the meter box is broken and wires are exposed", 2)],
        "hi": [("बिजली बिल्कुल नहीं है", 2), ("बिजली का तार टूटकर सड़क पर गिरा हुआ है", 3),
               ("ट्रांसफार्मर से चिंगारी निकल रही है", 3), ("रोज़ बार-बार बिजली कटौती होती है", 1),
               ("वोल्टेज ऊपर-नीचे हो रहा है, सामान खराब हो रहे हैं", 1), ("बिजली का बिल गलत आया है", 0),
               ("बिजली का खंभा खतरनाक तरीके से झुका हुआ है", 2)],
        "te": [("కరెంట్ అసలు లేదు", 2), ("కరెంట్ తీగ తెగి రోడ్డు మీద పడి ఉంది", 3),
               ("ట్రాన్స్‌ఫార్మర్ నుండి నిప్పురవ్వలు వస్తున్నాయి", 3), ("రోజూ పదే పదే కరెంట్ కోతలు ఉంటున్నాయి", 1),
               ("వోల్టేజ్ హెచ్చుతగ్గులతో ఇంట్లో వస్తువులు పాడవుతున్నాయి", 1), ("కరెంట్ బిల్లు తప్పుగా వచ్చింది", 0),
               ("కరెంట్ స్తంభం ప్రమాదకరంగా వంగి ఉంది", 2)],
        "ur": [("بجلی بالکل نہیں ہے", 2), ("بجلی کا تار ٹوٹ کر سڑک پر گرا ہوا ہے", 3),
               ("ٹرانسفارمر سے چنگاری نکل رہی ہے", 3), ("روزانہ بار بار بجلی کٹوتی ہوتی ہے", 1),
               ("بجلی کا بل غلط آیا ہے", 0)],
        "hinglish": [("bijli bilkul nahi hai", 2), ("bijli ka taar toot ke road pe gira hua hai", 3),
                     ("transformer se chingari nikal rahi hai", 3), ("roz baar baar power cut hota hai", 1),
                     ("voltage upar neeche ho raha hai, saaman kharab ho rahe hain", 1),
                     ("bijli ka bill galat aaya hai", 0)],
        "tenglish": [("current asalu ledu", 2), ("current teega thegi road meeda padi undi", 3),
                     ("transformer nundi nippuravvalu vasthunnayi", 3), ("roju pade pade current cuts untunnayi", 1),
                     ("voltage fluctuation valla intlo vasthuvulu paadavthunnayi", 1),
                     ("current bill thappuga vachindi", 0)],
    },
    "streetlights": {
        "en": [("the streetlights are not working", 1), ("the whole lane is completely dark at night", 2),
               ("the streetlight keeps flickering", 0), ("streetlights are on even during the day", 0),
               ("the streetlight pole is broken", 1),
               ("it is unsafe for women to walk because there are no lights", 2)],
        "hi": [("स्ट्रीट लाइट काम नहीं कर रही हैं", 1), ("रात को पूरी गली में अंधेरा रहता है", 2),
               ("स्ट्रीट लाइट बार-बार टिमटिमा रही है", 0), ("दिन में भी स्ट्रीट लाइट जलती रहती है", 0),
               ("स्ट्रीट लाइट का खंभा टूटा हुआ है", 1),
               ("लाइट न होने से महिलाओं को रात में चलने में डर लगता है", 2)],
        "te": [("వీధి దీపాలు పనిచేయడం లేదు", 1), ("రాత్రిపూట వీధి మొత్తం చీకటిగా ఉంటోంది", 2),
               ("స్ట్రీట్ లైట్ మినుకు మినుకుమంటోంది", 0), ("పగలు కూడా స్ట్రీట్ లైట్లు వెలుగుతూనే ఉన్నాయి", 0),
               ("స్ట్రీట్ లైట్ స్తంభం విరిగిపోయింది", 1),
               ("లైట్లు లేక ఆడవాళ్లు రాత్రి నడవడానికి భయపడుతున్నారు", 2)],
        "ur": [("اسٹریٹ لائٹس کام نہیں کر رہی ہیں", 1), ("رات کو پوری گلی میں اندھیرا رہتا ہے", 2),
               ("اسٹریٹ لائٹ بار بار ٹمٹما رہی ہے", 0), ("دن میں بھی اسٹریٹ لائٹ جلتی رہتی ہے", 0),
               ("روشنی نہ ہونے سے خواتین کو رات میں چلنے میں ڈر لگتا ہے", 2)],
        "hinglish": [("street lights kaam nahi kar rahi", 1), ("raat ko poori gali mein andhera rehta hai", 2),
                     ("street light baar baar timtima rahi hai", 0), ("din mein bhi street light jalti rehti hai", 0),
                     ("light na hone se ladies ko raat mein chalne mein darr lagta hai", 2)],
        "tenglish": [("street lights pani cheyyatledu", 1), ("raathri veedhi motham cheekatiga untondi", 2),
                     ("street light minuku minukumantondi", 0), ("pagalu kooda street lights veluguthune unnayi", 0),
                     ("lights leka aadavallu raathri nadavadaniki bhayapaduthunnaru", 2)],
    },
    "garbage": {
        "en": [("garbage has not been collected", 1), ("a huge garbage dump is rotting on the roadside", 2),
               ("people are burning garbage and the smoke is unbearable", 2), ("the garbage bin is overflowing", 1),
               ("a dead animal carcass is lying on the road", 2), ("the sweeper has not come to clean the street", 0),
               ("construction debris has been dumped on the footpath", 0)],
        "hi": [("कचरा उठाया नहीं गया है", 1), ("सड़क किनारे कचरे का बड़ा ढेर सड़ रहा है", 2),
               ("लोग कचरा जला रहे हैं, धुआँ सहन नहीं होता", 2), ("कूड़ेदान भरकर बाहर गिर रहा है", 1),
               ("सड़क पर मरे हुए जानवर की लाश पड़ी है", 2), ("सफाई कर्मचारी गली साफ करने नहीं आया", 0),
               ("फुटपाथ पर मलबा डाल दिया गया है", 0)],
        "te": [("చెత్త సేకరించడం లేదు", 1), ("రోడ్డు పక్కన పెద్ద చెత్త కుప్ప కుళ్ళిపోతోంది", 2),
               ("చెత్తను కాలుస్తున్నారు, పొగ భరించలేకపోతున్నాం", 2), ("చెత్త డబ్బా నిండిపోయి బయట పడుతోంది", 1),
               ("రోడ్డు మీద చచ్చిన జంతువు కళేబరం పడి ఉంది", 2), ("స్వీపర్ వీధి శుభ్రం చేయడానికి రాలేదు", 0),
               ("ఫుట్‌పాత్ మీద భవన నిర్మాణ వ్యర్థాలు వేశారు", 0)],
        "ur": [("کچرا اٹھایا نہیں گیا ہے", 1), ("سڑک کنارے کچرے کا بڑا ڈھیر سڑ رہا ہے", 2),
               ("کوڑے دان بھر کر باہر گر رہا ہے", 1), ("سڑک پر مرے ہوئے جانور کی لاش پڑی ہے", 2),
               ("صفائی کرمچاری گلی صاف کرنے نہیں آیا", 0)],
        "hinglish": [("kachra uthaya nahi gaya hai", 1), ("road ke kinare kachre ka bada dher sad raha hai", 2),
                     ("log kachra jala rahe hain, dhuan saha nahi jata", 2), ("dustbin bhar ke bahar gir raha hai", 1),
                     ("road pe mare hue jaanwar ki laash padi hai", 2), ("safai wala gali saaf karne nahi aaya", 0)],
        "tenglish": [("chetta collect cheyyatledu", 1), ("road pakkana pedda chetta kuppa kullipothondi", 2),
                     ("chettanu kaalusthunnaru, poga bharinchalekapothunnam", 2),
                     ("chetta dabba nindipoyi bayata paduthondi", 1),
                     ("road meeda chachina janthuvu kalebaram padi undi", 2),
                     ("sweeper veedhi clean cheyyadaniki raaledu", 0)],
    },
    "stray_animals": {
        "en": [("stray dogs are chasing people at night", 2), ("a stray dog bit a child", 3),
               ("a pack of aggressive dogs is roaming in the lane", 2),
               ("stray cattle are sitting in the middle of the road", 1),
               ("monkeys are entering houses and troubling people", 1), ("street dogs keep barking all night", 0),
               ("a dog with rabies symptoms is roaming around", 3)],
        "hi": [("आवारा कुत्ते रात में लोगों का पीछा करते हैं", 2), ("एक आवारा कुत्ते ने बच्चे को काट लिया", 3),
               ("गली में खूंखार कुत्तों का झुंड घूम रहा है", 2), ("आवारा गायें सड़क के बीच में बैठी रहती हैं", 1),
               ("बंदर घरों में घुसकर परेशान कर रहे हैं", 1), ("गली के कुत्ते पूरी रात भौंकते रहते हैं", 0)],
        "te": [("వీధి కుక్కలు రాత్రిపూట జనాలను వెంబడిస్తున్నాయి", 2), ("ఒక వీధి కుక్క పిల్లవాడిని కరిచింది", 3),
               ("వీధిలో భయంకరమైన కుక్కల గుంపు తిరుగుతోంది", 2), ("పశువులు రోడ్డు మధ్యలో కూర్చుంటున్నాయి", 1),
               ("కోతులు ఇళ్లలోకి వచ్చి ఇబ్బంది పెడుతున్నాయి", 1), ("వీధి కుక్కలు రాత్రంతా మొరుగుతూనే ఉన్నాయి", 0)],
        "ur": [("آوارہ کتے رات میں لوگوں کا پیچھا کرتے ہیں", 2), ("ایک آوارہ کتے نے بچے کو کاٹ لیا", 3),
               ("گلی میں خطرناک کتوں کا جھنڈ گھوم رہا ہے", 2), ("آوارہ مویشی سڑک کے بیچ میں بیٹھے رہتے ہیں", 1),
               ("بندر گھروں میں گھس کر پریشان کر رہے ہیں", 1)],
        "hinglish": [("awaara kutte raat mein logon ka peecha karte hain", 2),
                     ("ek awaara kutte ne bachche ko kaat liya", 3),
                     ("gali mein khoonkhar kutton ka jhund ghoom raha hai", 2),
                     ("awaara gaayein road ke beech mein baithi rehti hain", 1),
                     ("bandar gharon mein ghus ke pareshaan kar rahe hain", 1),
                     ("gali ke kutte poori raat bhaunkte rehte hain", 0)],
        "tenglish": [("veedhi kukkalu raathri janalni vembadisthunnayi", 2),
                     ("oka veedhi kukka pillavadini karichindi", 3),
                     ("veedhilo bhayankaramaina kukkala gumpu thiruguthondi", 2),
                     ("pashuvulu road madhyalo koorchuntunnayi", 1),
                     ("kothulu illaloki vachi ibbandi pedutunnayi", 1),
                     ("veedhi kukkalu raathrantha moruguthune unnayi", 0)],
    },
    "mosquito": {
        "en": [("there is a severe mosquito menace", 1), ("many people in our colony have dengue fever", 2),
               ("stagnant water has become a mosquito breeding ground", 1), ("fogging has not been done for months", 1),
               ("the empty plot is full of bushes and mosquitoes", 0), ("children are falling sick with malaria", 2)],
        "hi": [("मच्छरों का बहुत ज़्यादा आतंक है", 1), ("हमारी कॉलोनी में कई लोगों को डेंगू हो गया है", 2),
               ("रुके हुए पानी में मच्छर पनप रहे हैं", 1), ("महीनों से फॉगिंग नहीं हुई है", 1),
               ("खाली प्लॉट में झाड़ियाँ और मच्छर भरे हैं", 0), ("बच्चे मलेरिया से बीमार पड़ रहे हैं", 2)],
        "te": [("దోమల బెడద చాలా ఎక్కువగా ఉంది", 1), ("మా కాలనీలో చాలా మందికి డెంగ్యూ జ్వరం వచ్చింది", 2),
               ("నిలిచిన నీటిలో దోమలు పెరుగుతున్నాయి", 1), ("నెలల తరబడి ఫాగింగ్ చేయలేదు", 1),
               ("ఖాళీ ప్లాట్ నిండా పొదలు, దోమలు ఉన్నాయి", 0), ("పిల్లలు మలేరియాతో అనారోగ్యం పాలవుతున్నారు", 2)],
        "ur": [("مچھروں کی بہت زیادہ بھرمار ہے", 1), ("ہماری کالونی میں کئی لوگوں کو ڈینگی ہو گیا ہے", 2),
               ("رکے ہوئے پانی میں مچھر پیدا ہو رہے ہیں", 1), ("مہینوں سے فوگنگ نہیں ہوئی ہے", 1),
               ("بچے ملیریا سے بیمار پڑ رہے ہیں", 2)],
        "hinglish": [("machharon ka bahut zyada aatank hai", 1), ("hamari colony mein kai logon ko dengue ho gaya hai", 2),
                     ("ruke hue paani mein machhar paida ho rahe hain", 1), ("mahino se fogging nahi hui hai", 1),
                     ("khaali plot mein jhaadiyan aur machhar bhare hain", 0)],
        "tenglish": [("domala bedada chala ekkuvaga undi", 1),
                     ("maa colony lo chala mandiki dengue jwaram vachindi", 2),
                     ("nilichina neetilo domalu peruguthunnayi", 1), ("nelala tharabadi fogging cheyyaledu", 1),
                     ("khaali plot ninda podalu, domalu unnayi", 0)],
    },
}

# Terse, keyword-style complaints ("pothole", "no current since morning") as people type them in apps.
# Train-only (never held out); label = base urgency + rules.
SHORT = {
    "roads": {"en": [("pothole on main road", 1), ("road damaged badly", 1), ("big pit on road, bikes falling", 2)],
              "te": [("రోడ్డు మీద గుంతలు", 1), ("రోడ్డు పూర్తిగా పాడైంది", 1)],
              "hi": [("सड़क पर गड्ढा", 1), ("सड़क बहुत खराब है", 1)], "ur": [("سڑک پر گڑھا", 1)],
              "hinglish": [("road pe gaddha", 1), ("road bahut kharab hai", 1)],
              "tenglish": [("road meeda guntha", 1), ("road antha paadaipoyindi", 1)]},
    "water": {"en": [("no water since morning", 2), ("tap water dirty", 2), ("water leakage from pipe", 1)],
              "te": [("నీళ్లు రావడం లేదు", 2), ("నల్లా నీళ్లు మురికిగా వస్తున్నాయి", 2)],
              "hi": [("पानी नहीं आ रहा", 2), ("नल का पानी गंदा", 2)], "ur": [("پانی نہیں آ رہا", 2)],
              "hinglish": [("paani nahi aa raha", 2), ("nal ka paani ganda hai", 2)],
              "tenglish": [("neellu raavatledu", 2), ("nalla neellu murikiga vasthunnayi", 2)]},
    "sewerage": {"en": [("drainage overflow", 2), ("drain blocked", 1), ("sewage smell everywhere", 1)],
                 "te": [("డ్రైనేజీ పొంగుతోంది", 2), ("డ్రైనేజీ బ్లాక్ అయింది", 1)],
                 "hi": [("नाला ओवरफ्लो", 2), ("नाली जाम है", 1)], "ur": [("نالہ ابل رہا ہے", 2)],
                 "hinglish": [("naala overflow ho raha", 2), ("naali block hai", 1)],
                 "tenglish": [("drainage overflow avthondi", 2), ("drainage block ayindi", 1)]},
    # In Hinglish/Hindi "light" usually means household power, not streetlights.
    "electricity": {"en": [("no current since morning", 2), ("power cut again", 1), ("transformer problem", 1),
                           ("no light in our house since evening", 2)],
                    "te": [("కరెంట్ పోయింది", 2), ("మళ్ళీ కరెంట్ కట్", 1), ("ఇంట్లో కరెంట్ లేదు", 2)],
                    "hi": [("बिजली गुल है", 2), ("फिर से लाइट कटी", 1), ("घर में लाइट नहीं है", 2)],
                    "ur": [("بجلی غائب ہے", 2), ("گھر میں لائٹ نہیں ہے", 2)],
                    "hinglish": [("light gayi hui hai", 2), ("phir se power cut", 1), ("ghar mein light nahi hai", 2),
                                 ("subah se light nahi hai", 2)],
                    "tenglish": [("current poyindi", 2), ("malli current cut", 1), ("intlo current ledu", 2)]},
    "streetlights": {"en": [("street light not working", 1), ("dark road no lights", 2)],
                     "te": [("స్ట్రీట్ లైట్ వెలగడం లేదు", 1), ("వీధి చీకటిగా ఉంది", 2)],
                     "hi": [("स्ट्रीट लाइट बंद है", 1), ("गली में अंधेरा", 2)], "ur": [("اسٹریٹ لائٹ بند ہے", 1)],
                     "hinglish": [("street light band hai", 1), ("gali mein andhera", 2)],
                     "tenglish": [("street light veligatledu", 1), ("veedhi cheekatiga undi", 2)]},
    "garbage": {"en": [("garbage not picked up", 1), ("trash dumped on road", 1), ("garbage smell", 1)],
                "te": [("చెత్త తీయడం లేదు", 1), ("రోడ్డు మీద చెత్త", 1)],
                "hi": [("कचरा नहीं उठा", 1), ("सड़क पर कूड़ा", 1)], "ur": [("کچرا نہیں اٹھا", 1)],
                "hinglish": [("kachra nahi utha", 1), ("road pe kooda pada hai", 1)],
                "tenglish": [("chetta theeyatledu", 1), ("road meeda chetta", 1)]},
    "stray_animals": {"en": [("stray dog problem", 1), ("dogs attacking people", 2), ("cows on road", 1)],
                      "te": [("కుక్కల బెడద", 1), ("కుక్కలు మీద పడుతున్నాయి", 2)],
                      "hi": [("आवारा कुत्तों की समस्या", 1), ("कुत्ते लोगों पर हमला कर रहे", 2)],
                      "ur": [("آوارہ کتوں کا مسئلہ", 1)],
                      "hinglish": [("kutton ka problem", 1), ("kutte logon pe attack kar rahe", 2)],
                      "tenglish": [("kukkala bedada", 1), ("kukkalu janala meeda paduthunnayi", 2)]},
    "mosquito": {"en": [("too many mosquitoes", 1), ("need fogging", 1), ("dengue cases in area", 2)],
                 "te": [("దోమలు ఎక్కువయ్యాయి", 1), ("ఫాగింగ్ చేయాలి", 1)],
                 "hi": [("मच्छर बहुत हैं", 1), ("फॉगिंग करवाइए", 1)], "ur": [("مچھر بہت ہیں", 1)],
                 "hinglish": [("machhar bahut hain", 1), ("fogging karwao", 1)],
                 "tenglish": [("domalu ekkuvayyayi", 1), ("fogging cheyyali", 1)]},
}

_SCRIPT_FOR = {"te": "te", "hi": "hi", "ur": "ur"}


def _loc_name(loc: dict, lang: str, rng: random.Random) -> str:
    want = _SCRIPT_FOR.get(lang)
    if want:
        native = [a for a in loc["aliases"] if script_of(a) == want]
        # people often write the area name in English even inside a Telugu/Hindi sentence
        if native and rng.random() > 0.15:
            return rng.choice(native)
        return loc["name"]
    latin = [a for a in loc["aliases"] if a.isascii()]
    return rng.choice([loc["name"], loc["name"], *latin])


def _typo(text: str, rng: random.Random) -> str:
    chars = list(text)
    for _ in range(rng.randint(1, 3)):
        if len(chars) < 6:
            break
        i = rng.randrange(1, len(chars) - 1)
        op = rng.random()
        if op < 0.4:
            del chars[i]
        elif op < 0.8:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        else:
            chars.insert(i, chars[i])
    return "".join(chars)


def holdout_split(seed: int = 7) -> dict[tuple[str, str], set[int]]:
    """Which issue clauses (by index) are reserved for the test split in each cell."""
    rng = random.Random(seed)
    held = {}
    for dept, by_lang in ISSUES.items():
        for lang, phrases in by_lang.items():
            k = 2 if len(phrases) >= 7 else 1
            held[(dept, lang)] = set(rng.sample(range(len(phrases)), k))
    return held


def make_short(dept: str, lang: str, rng: random.Random, localities: list[dict]) -> dict:
    issue, base = rng.choice(SHORT[dept][lang])
    loc = rng.choice(localities) if rng.random() > 0.3 else None
    text = issue
    if loc:
        name = _loc_name(loc, lang, rng)
        text = rng.choice([f"{issue} - {name}", f"{name}: {issue}", f"{issue} {LOC_FORMS[lang].format(name=name)}"])
    if lang in ("en", "tenglish", "hinglish") and rng.random() < 0.15:
        text = _typo(text, rng)
    label = rules.apply(config.urgency_ids()[base], rules.check(text))
    return {"text": text, "department": dept, "urgency": label, "lang": lang,
            "locality": loc["id"] if loc else "", "phrase": f"{dept}/{lang}/short"}


def make_one(dept: str, lang: str, phrase_idx: int, rng: random.Random, localities: list[dict],
             noise: float = 0.15) -> dict:
    issue, base = ISSUES[dept][lang][phrase_idx]
    durs, extras = DURATIONS[lang], EXTRAS[lang]
    # context that raises urgency appears in roughly a quarter of complaints each
    dur, dbump = rng.choices(durs, weights=[3 if b == 0 else 1.2 for _, b in durs])[0]
    extra, ebump = rng.choices(extras, weights=[3 if b == 0 else 1.2 for _, b in extras])[0]
    loc = rng.choice(localities) if rng.random() > 0.1 else None
    loc_phrase = LOC_FORMS[lang].format(name=_loc_name(loc, lang, rng)) if loc else NO_LOC[lang]
    frame = rng.choice(FRAMES[lang])
    text = frame.format(issue=issue, loc=loc_phrase, dur=dur, extra=extra)
    text = " ".join(text.split())
    text = text[0].upper() + text[1:]

    if lang in ("en", "tenglish", "hinglish"):
        if rng.random() < noise:
            text = _typo(text, rng)
        if rng.random() < 0.3:
            text = text.lower()
        if rng.random() < 0.2:
            text = text.replace(".", "").replace(",", "")

    order = config.urgency_ids()
    level = base + min(1, dbump + ebump)  # context adds at most one level
    level = min(level, 3) if base == 3 else min(level, 2)
    label = rules.apply(order[level], rules.check(text))
    return {"text": text, "department": dept, "urgency": label, "lang": lang,
            "locality": loc["id"] if loc else "", "phrase": f"{dept}/{lang}/{phrase_idx}"}


def generate(n: int, split: str, seed: int = 0) -> list[dict]:
    """split: 'train' (non-held-out clauses) or 'test' (held-out clauses only)."""
    rng = random.Random(seed)
    held = holdout_split()
    localities = config.city()["localities"]
    cells = []
    for dept, by_lang in ISSUES.items():
        for lang, phrases in by_lang.items():
            idx = [i for i in range(len(phrases)) if (i in held[(dept, lang)]) == (split == "test")]
            cells.extend((dept, lang, i) for i in idx)
    rows = []
    for _ in range(n):
        if split == "train" and rng.random() < 0.12:
            dept, lang = rng.choice(list(SHORT)), rng.choice(LANGS)
            rows.append(make_short(dept, lang, rng, localities))
            continue
        dept, lang, i = rng.choice(cells)
        rows.append(make_one(dept, lang, i, rng, localities))
    return rows
