"""
Agricultural Disease Advisory Module
AI-Enabled Crop Disease Detection and Advisory System

Provides comprehensive agricultural agronomic advisory for all 15 supported crop conditions:
- Common Symptoms
- Cultural and Biological Prevention
- Safe Non-Chemical / Integrated Pest Management Suggestions
- Preliminary Diagnostic Disclaimer
"""

ADVISORY_DATABASE = {
    "Pepper__bell___Bacterial_spot": {
        "crop": "Pepper (Bell Pepper)",
        "condition": "Bacterial Spot",
        "scientific_name": "Xanthomonas campestris pv. vesicatoria",
        "category": "diseased",
        "symptoms": [
            "Small, water-soaked, circular to irregular lesions on leaves that turn dark brown to black.",
            "Leaf spots often develop a yellow chlorotic halo.",
            "Severe infections lead to extensive defoliation, leaving fruits exposed to sunscald.",
            "Raised, blister-like or scabby brown spots on fruit surface."
        ],
        "prevention": [
            "Use certified disease-free, pathogen-tested seeds and transplants.",
            "Avoid overhead irrigation; use drip irrigation to minimize leaf wetness.",
            "Implement a 2- to 3-year crop rotation avoiding other solanaceous crops (tomatoes, potatoes, eggplants).",
            "Sanitize tools, stakes, and footwear between field sections."
        ],
        "management": [
            "Promptly rogue and destroy heavily infected plants and crop residues.",
            "Apply approved preventative copper-based bactericides or bio-fungicides (e.g., Bacillus subtilis) according to agricultural extension guidelines.",
            "Avoid working in fields when foliage is wet to prevent bacterial spreading."
        ]
    },
    "Pepper__bell___healthy": {
        "crop": "Pepper (Bell Pepper)",
        "condition": "Healthy Plant",
        "scientific_name": "Capsicum annuum",
        "category": "healthy",
        "symptoms": [
            "Vibrant green, uniform foliage with smooth leaf margins.",
            "No visible lesions, chlorosis, necrosis, or wilting.",
            "Strong stem architecture and vigorous flowering or fruiting."
        ],
        "prevention": [
            "Maintain balanced N-P-K fertilization and adequate calcium to prevent blossom end rot.",
            "Ensure well-drained soil with pH between 6.2 and 6.8.",
            "Inspect weekly for early signs of aphids, thrips, or whiteflies."
        ],
        "management": [
            "Continue standard agronomic practices and maintain consistent soil moisture."
        ]
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "condition": "Early Blight",
        "scientific_name": "Alternaria solani",
        "category": "diseased",
        "symptoms": [
            "Characteristic concentric circular rings creating a target-board or bullseye pattern.",
            "Starts on older, lower foliage and gradually moves upward.",
            "Lesions turn dark brown and surrounding tissue becomes chlorotic (yellow).",
            " premature defoliation and tuber quality degradation."
        ],
        "prevention": [
            "Plant certified disease-free seed tubers of tolerant or resistant varieties.",
            "Maintain plant vigor with balanced nitrogen; nitrogen-deficient plants are more susceptible.",
            "Practice 3-year crop rotation with non-host crops like cereals or legumes.",
            "Allow proper vine senescence before harvest."
        ],
        "management": [
            "Remove infected lower leaves early in the season to reduce inoculum pressure.",
            "Apply preventative protectant bio-fungicides or recommended contact fungicides during warm, humid weather.",
            "Ensure good field airflow through appropriate plant spacing."
        ]
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "condition": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "category": "diseased",
        "symptoms": [
            "Water-soaked, rapidly expanding pale to dark lesions on leaf tips or margins.",
            "White, velvety fungal-like downy growth visible on leaf undersides in humid or wet conditions.",
            "Rapid collapse and browning of entire leaf and stem tissues ('blight').",
            "Tubers develop dry, firm, reddish-brown granular rot beneath the skin."
        ],
        "prevention": [
            "Eliminate cull piles, volunteer potato plants, and infected tubers prior to the season.",
            "Use certified clean seed tubers and blight-resistant cultivars where available.",
            "Avoid sprinkler irrigation; irrigate early in the morning so foliage dries rapidly.",
            "Monitor local late-blight forecast alerts and weather conditions closely."
        ],
        "management": [
            "Destroy localized infection pockets immediately to prevent field-wide epidemic spread.",
            "Apply registered systemic and protective fungicides under university or agricultural department advisory.",
            "Desiccate or kill vines at least two weeks prior to harvest to prevent tuber contamination during lifting."
        ]
    },
    "Potato___healthy": {
        "crop": "Potato",
        "condition": "Healthy Plant",
        "scientific_name": "Solanum tuberosum",
        "category": "healthy",
        "symptoms": [
            "Lush, dark-green pinnate foliage without spots or blights.",
            "Robust upright stem development and healthy root system.",
            "Absence of wilting, discoloration, or leaf curling."
        ],
        "prevention": [
            "Ensure regular hilling to protect developing tubers from sunlight (greening) and tuber moths.",
            "Supply adequate soil potassium and phosphorus during tuber bulking.",
            "Keep soil evenly moist without waterlogging."
        ],
        "management": [
            "Maintain current irrigation and pest-monitoring schedule."
        ]
    },
    "Tomato_Bacterial_spot": {
        "crop": "Tomato",
        "condition": "Bacterial Spot",
        "scientific_name": "Xanthomonas spp.",
        "category": "diseased",
        "symptoms": [
            "Small, dark brown to black water-soaked spots (1-3 mm) on leaves.",
            "Lesions may coalesce, causing leaflets to turn brown and wither.",
            "Fruit exhibits small, raised, black blister-like spots that become scab-like with white halos.",
            "Severe defoliation exposing green tomatoes to solar injury."
        ],
        "prevention": [
            "Use hot-water treated or certified disease-free certified seeds.",
            "Never work or prune in tomato rows while plants are wet with dew or rain.",
            "Sterilize trellising stakes, ties, and clipping shears between seasons.",
            "Avoid overhead irrigation to minimize bacterial splash dispersal."
        ],
        "management": [
            "Remove and incinerate or deeply bury infected plant debris after harvest.",
            "Apply preventative copper sprays mixed with ethylene bisdithiocarbamate (EBDC) or biocontrol agents where approved.",
            "Rotate tomato plots out of Solanaceae for at least 2 seasons."
        ]
    },
    "Tomato_Early_blight": {
        "crop": "Tomato",
        "condition": "Early Blight",
        "scientific_name": "Alternaria linariae (Alternaria solani)",
        "category": "diseased",
        "symptoms": [
            "Dark brown concentric ring spots ('target board') on older leaves.",
            "Yellowing of surrounding tissue and gradual leaf drop from bottom upwards.",
            "Dark, sunken, leathery lesions on stems at soil level ('collar rot' in seedlings).",
            "Stem-end sunken rot on developing fruit."
        ],
        "prevention": [
            "Apply organic mulch (straw or plastic) around plant base to prevent soil-borne spore splashing.",
            "Maintain adequate spacing (60-90 cm) and prune lower suckers to promote air circulation.",
            "Stake or cage indeterminate vines to keep foliage off the ground.",
            "Follow 3-year crop rotation schedules."
        ],
        "management": [
            "Prune off lower infected leaves as soon as first symptoms appear; disinfect pruners.",
            "Apply protective organic sprays such as copper octanoate, neem formulations, or biofungicides.",
            "Avoid overhead watering; water strictly at ground level."
        ]
    },
    "Tomato_Late_blight": {
        "crop": "Tomato",
        "condition": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "category": "diseased",
        "symptoms": [
            "Large, irregularly shaped greasy gray-green or brown water-soaked lesions.",
            "White fungal mold growth on the underside of affected leaves in cool, moist conditions.",
            "Rapid stem browning and sudden foliage collapse, resembling frost damage.",
            "Firm, brown, bumpy lesions with greasy appearance on green fruits."
        ],
        "prevention": [
            "Plant resistant tomato hybrids (e.g., Defiant, Mountain Magic, Iron Lady).",
            "Do not plant tomatoes in close proximity to potato plantings.",
            "Ensure rapid canopy drying by choosing sunny planting locations with good breeze.",
            "Sanitize all stakes and wire supports."
        ],
        "management": [
            "Act quickly: pull out and bag entire infected plants; do not compost late blight debris.",
            "Notify neighboring growers if late blight is confirmed in your locality.",
            "Apply protective registered fungicides before rainy, cool periods occur."
        ]
    },
    "Tomato_Leaf_Mold": {
        "crop": "Tomato",
        "condition": "Leaf Mold",
        "scientific_name": "Passalora fulva (Fulvia fulva)",
        "category": "diseased",
        "symptoms": [
            "Pale green to yellow spots with indefinite margins on upper leaf surfaces.",
            "Olive green, velvety or brown fungal down on the corresponding leaf undersides.",
            "Heavily affected leaves roll upward, wither, and drop prematurely.",
            "Most prominent in high humidity, greenhouses, high tunnels, or crowded hoop houses."
        ],
        "prevention": [
            "Maximize greenhouse or canopy ventilation and lower relative humidity below 85%.",
            "Increase plant spacing and prune lower foliage to encourage continuous airflow.",
            "Use drip irrigation exclusively; do not wet tomato leaves.",
            "Select resistant tomato cultivars containing Cf-resistance genes."
        ],
        "management": [
            "Prune off affected foliage during dry hours.",
            "Wash and steam-sanitize greenhouse structures between crop cycles.",
            "Apply authorized bio-fungicides (e.g., Trichoderma or Bacillus species) to prevent mold sporulation."
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "crop": "Tomato",
        "condition": "Septoria Leaf Spot",
        "scientific_name": "Septoria lycopersici",
        "category": "diseased",
        "symptoms": [
            "Numerous small (2-4 mm), circular spots with gray or tan centers and dark brown margins.",
            "Tiny black specks (pycnidia - spore-bearing structures) visible in the center of mature spots.",
            "Causes severe yellowing and premature dropping of lower foliage.",
            "Fruit is rarely directly infected, but sunscald occurs due to defoliation."
        ],
        "prevention": [
            "Mulch heavily beneath plants to create a physical barrier against soil-splashed spores.",
            "Water only at the base with soaker hoses or drip tape.",
            "Remove all solanaceous weeds (such as nightshades and horsenettle) around the garden perimeter.",
            "Practice 2- to 3-year crop rotations."
        ],
        "management": [
            "Remove infected lower leaves promptly at first sign; discard in sealed trash bags.",
            "Apply broad-spectrum preventative fungicides or approved copper octanoate at 7-10 day intervals during wet periods.",
            "Thoroughly clean up and burn or bury fallen crop debris in autumn."
        ]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "crop": "Tomato",
        "condition": "Spider Mites (Two-spotted spider mite)",
        "scientific_name": "Tetranychus urticae",
        "category": "diseased",
        "symptoms": [
            "Fine yellow or white stippling (tiny dots) on the upper leaf surface.",
            "Delicate silken webbing visible on leaf undersides, growing tips, and between stems.",
            "Leaves turn bronze, dry, brittle, and drop off in severe infestations.",
            "Mites are tiny (0.5 mm) oval arachnids visible under a 10x hand lens."
        ],
        "prevention": [
            "Avoid excessive nitrogen fertilization, which stimulates tender mite-favored growth.",
            "Keep farm roads and field perimeters moist to reduce dusty conditions that favor mites.",
            "Preserve and encourage natural predators (lady beetles, predatory mites, lacewings).",
            "Avoid broad-spectrum pyrethroid insecticides that destroy beneficial predatory mites."
        ],
        "management": [
            "Knock down populations with high-pressure water spray on leaf undersides.",
            "Release biological control agents such as predatory mites (Phytoseiulus persimilis).",
            "Apply insecticidal soap, horticultural mineral oils, or azadirachtin (neem extract) with thorough coverage."
        ]
    },
    "Tomato__Target_Spot": {
        "crop": "Tomato",
        "condition": "Target Spot",
        "scientific_name": "Corynespora cassiicola",
        "category": "diseased",
        "symptoms": [
            "Small, pinpoint lesions that enlarge into round brown spots with concentric rings.",
            "Spots usually have light brown centers with darker margins and yellow halos.",
            "Lesions on fruit are circular, deeply sunken, and develop dark brown centers.",
            "Severe infections can lead to rapid blighting of upper canopy leaves."
        ],
        "prevention": [
            "Ensure proper drainage and avoid water stagnation around root zones.",
            "Space plants adequately to facilitate quick drying of leaves after rains.",
            "Keep tomato plantings isolated from alternate hosts (cotton, soybean, cucumber).",
            "Use mulch to prevent splash dissemination of soil-dwelling inoculum."
        ],
        "management": [
            "Prune and safely discard diseased foliage.",
            "Apply protective fungicides (such as copper hydroxide or azoxystrobin where labeled).",
            "Clean and sanitize all stakes, support twine, and garden tools after each crop cycle."
        ]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "crop": "Tomato",
        "condition": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "scientific_name": "Begomovirus (TYLCV)",
        "category": "diseased",
        "symptoms": [
            "Severe upward and inward cupping and curling of leaf margins.",
            "Prominent interveinal chlorosis (yellowing) with stunted, deformed leaflets.",
            "Marked plant stunting and bushiness with very short internodes.",
            "Heavy flower abscission; plants infected early bear virtually no harvestable fruit."
        ],
        "prevention": [
            "Plant TYLCV-resistant or TYLCV-tolerant tomato varieties (e.g., Tycoon, Camaro).",
            "Use physical insect-proof netting (50-mesh screens) in nursery and greenhouse settings.",
            "Install yellow sticky traps to detect and monitor whitefly (Bemisia tabaci) populations early.",
            "Maintain a host-free crop break period between consecutive growing cycles."
        ],
        "management": [
            "Immediately pull up, bag, and destroy virus-infected plants; viruses cannot be cured.",
            "Target vector control: manage silverleaf whiteflies with insecticidal soaps, neem oil, or biological agents (Beauveria bassiana).",
            "Remove weed hosts (e.g., Datura, Solanum nigrum, Malva) that harbor the virus and vector."
        ]
    },
    "Tomato__Tomato_mosaic_virus": {
        "crop": "Tomato",
        "condition": "Tomato Mosaic Virus (ToMV)",
        "scientific_name": "Tobamovirus (ToMV)",
        "category": "diseased",
        "symptoms": [
            "Mottling of leaves with alternating light green and dark green mosaic patterns.",
            "Leaf distortion, blistering, crinkling, and 'shoestring' or 'fern-leaf' deformity.",
            "General plant stunting and delayed ripening with internal fruit browning (brown wall).",
            "Extremely stable virus transmitted mechanically via hands, shears, and clothing."
        ],
        "prevention": [
            "Plant ToMV-resistant tomato hybrids (look for 'T' or 'ToMV' disease code on seed packets).",
            "Wash hands thoroughly with soap or milk solution before handling tomato plants.",
            "Do not smoke or use tobacco products near tomato gardens (tobacco carries related viruses).",
            "Disinfect pruning shears in a 10-20% household bleach or trisodium phosphate (TSP) solution."
        ],
        "management": [
            "Rogue out and destroy infected plants immediately upon symptom detection.",
            "Do not cultivate or prune adjacent healthy plants after touching an infected specimen.",
            "Deeply till or solarize beds after harvest to speed decomposition of infected root residue."
        ]
    },
    "Tomato_healthy": {
        "crop": "Tomato",
        "condition": "Healthy Plant",
        "scientific_name": "Solanum lycopersicum",
        "category": "healthy",
        "symptoms": [
            "Rich deep green, fully expanded compound leaves without spots, necrosis, or curling.",
            "Vigorous stem growth, active terminal bud development, and healthy blossom clusters.",
            "Uniform fruit set without blemishes, cracking, or blossom end rot."
        ],
        "prevention": [
            "Provide consistent 1-1.5 inches of water per week through drip or ground irrigation.",
            "Maintain soil mulch and consistent moisture to prevent blossom end rot and fruit splitting.",
            "Perform weekly scouting on leaf undersides for early detection of pests or foliar spots."
        ],
        "management": [
            "Continue recommended fertilization, pruning, and trellising schedules."
        ]
    }
}

DISCLAIMER_TEXT = (
    "DISCLAIMER: This assessment is generated by an Artificial Intelligence deep learning "
    "model as an educational and preliminary diagnostic tool. Agricultural conditions can be "
    "influenced by complex environmental factors, co-infections, and soil nutrition. "
    "Before undertaking major chemical treatments or economic interventions, please consult your "
    "local agricultural extension service, certified plant pathologist, or agronomy expert."
)

def get_advisory_for_class(class_name: str) -> dict:
    """Returns the comprehensive advisory dictionary for a given class folder name."""
    info = ADVISORY_DATABASE.get(class_name)
    if not info:
        return {
            "crop": "Unknown",
            "condition": class_name,
            "category": "unknown",
            "symptoms": ["Detailed symptom information not available for this class."],
            "prevention": ["Maintain standard agricultural sanitation and hygiene."],
            "management": ["Consult a local agricultural extension specialist."],
            "disclaimer": DISCLAIMER_TEXT
        }

    advisory_payload = dict(info)
    advisory_payload["disclaimer"] = DISCLAIMER_TEXT
    return advisory_payload
