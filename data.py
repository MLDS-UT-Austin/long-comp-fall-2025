from enum import Enum

# This file contains useful data for agents to use ###############################


# All possible locations in the game
class Location(Enum):
    BLANTON_MUSEUM = "Blanton Museum"
    HARRY_RANSOM_CENTER = "Harry Ransom Center"
    LBJ_LIBRARY = "LBJ Library"
    FLAWN_ACADEMIC_CENTER = "Flawn Academic Center"
    ART_BUILDING_AND_MUSEUM = "Art Building and Museum"
    GATES_DELL_COMPLEX = "Gates Dell Complex"
    CHRISTIAN_GREEN_GALLERY = "Christian Green Gallery"
    FINE_ARTS_LIBRARY = "Fine Arts Library"
    DARRELL_K_ROYAL_TEXAS_MEMORIAL_STADIUM = "Darrell K Royal Texas Memorial Stadium"
    GREGORY_GYM = "Gregory Gym"
    TEXAS_MEMORIAL_MUSEUM = "Texas Memorial Museum"
    UT_TOWER = "UT Tower"
    LITTLEFIELD_FOUNTAIN = "Littlefield Fountain"
    TURTLE_POND = "Turtle Pond"
    THE_DRAG = "The Drag"
    NORMAN_HACKERMAN_BUILDING = "Norman Hackerman Building"
    WCP_STUDENT_ACTIVITY_CENTER = "WCP Student Activity Center"
    TEXAS_STATE_CAPITOL = "Texas State Capitol"
    MOUNT_BONNELL_AND_MAYFIELD_PARK = "Mount Bonnell & Mayfield Park"
    BARTON_SPRINGS = "Barton Springs"
    ZILKER_PARK = "Zilker Park"
    LAKE_AUSTIN = "Lake Austin"
    LADY_BIRD_JOHNSON_WILDFLOWER_CENTER = "Lady Bird Johnson Wildflower Center"
    ZILKER_BOTANICAL_GARDEN = "Zilker Botanical Garden"
    CONGRESS_AVENUE_BRIDGE = "Congress Avenue Bridge"
    SOUTH_CONGRESS = "South Congress"


# A dictionary that can optionally be used by agents to redact LLM output based on the location
# fmt: off
redaction_dict = {
    Location.BLANTON_MUSEUM: ["blanton", "museum", "art", "gallery", "exhibit", "painting", "curator"],
    Location.HARRY_RANSOM_CENTER: ["ransom center", "harry ransom", "archive", "manuscript", "collection", "literature", "photography"],
    Location.LBJ_LIBRARY: ["lbj", "library", "museum", "presidential", "johnson", "exhibit", "archives"],
    Location.FLAWN_ACADEMIC_CENTER: ["flawn", "academic center", "study room", "tutoring", "library", "student"],
    Location.ART_BUILDING_AND_MUSEUM: ["art building", "museum", "studio", "exhibit", "painting", "sculpture"],
    Location.GATES_DELL_COMPLEX: ["gates dell complex", "gdc", "computer science", "programming", "lab", "coding", "engineering"],
    Location.CHRISTIAN_GREEN_GALLERY: ["christian-green", "gallery", "art", "exhibit", "curator", "installation"],
    Location.FINE_ARTS_LIBRARY: ["fine arts library", "library", "books", "research", "fine arts", "study", "quiet zone"],
    Location.DARRELL_K_ROYAL_TEXAS_MEMORIAL_STADIUM: ["darrell k royal", "stadium", "football", "longhorns", "game day", "fans", "scoreboard"],
    Location.GREGORY_GYM: ["gregory gym", "gym", "workout", "weights", "basketball", "pool", "fitness"],
    Location.TEXAS_MEMORIAL_MUSEUM: ["texas memorial museum", "museum", "dinosaur", "fossil", "exhibit", "science", "natural history"],
    Location.UT_TOWER: ["ut tower", "tower", "university landmark", "campus", "clock tower", "observation deck"],
    Location.LITTLEFIELD_FOUNTAIN: ["littlefield fountain", "fountain", "statue", "memorial", "landmark", "water feature"],
    Location.TURTLE_POND: ["turtle pond", "pond", "turtle", "wildlife", "water", "nature", "quiet area"],
    Location.THE_DRAG: ["the drag", "guadalupe", "shops", "street", "café", "restaurant", "thrift", "traffic"],
    Location.NORMAN_HACKERMAN_BUILDING: ["norman hackerman building", "nhb", "chemistry", "research", "lab", "science", "lecture"],
    Location.WCP_STUDENT_ACTIVITY_CENTER: ["wcp", "student activity center", "union", "study", "event", "meeting", "food court"],
    Location.TEXAS_STATE_CAPITOL: ["texas state capitol", "capitol", "government", "legislature", "governor", "building", "austin landmark"],
    Location.MOUNT_BONNELL_AND_MAYFIELD_PARK: ["mt bonnell", "mayfield park", "peacock", "hike", "viewpoint", "scenic", "trail"],
    Location.BARTON_SPRINGS: ["barton springs", "pool", "spring", "swim", "bathing", "cold water", "zilker"],
    Location.ZILKER_PARK: ["zilker park", "zilker", "park", "picnic", "festival", "trail", "outdoors"],
    Location.LAKE_AUSTIN: ["lake austin", "lake", "boat", "kayak", "water", "dock", "scenic"],
    Location.LADY_BIRD_JOHNSON_WILDFLOWER_CENTER: ["wildflower center", "lady bird johnson", "wildflower", "garden", "native plants", "botanical"],
    Location.ZILKER_BOTANICAL_GARDEN: ["zilker botanical garden", "botanical", "garden", "plants", "flowers", "greenhouse", "pond"],
    Location.CONGRESS_AVENUE_BRIDGE: ["congress avenue bridge", "bridge", "bats", "downtown", "austin skyline", "river"],
    Location.SOUTH_CONGRESS: ["south congress", "soco", "shopping", "restaurants", "street", "murals", "austin"],
}

# fmt: on

# Everthing below is for internal use to generate the game dialogue #####################################

SPY_REVEAL_AND_GUESS = (
    "Muah ha ha! I was the spy all along! Was it the {location}?",
    "Jokes on you all! I was the spy! Was the {location}.",
    "You never suspected me, did you? I was right under your noses! Was it the {location}?",
    "Congratulations, you’ve played right into my hands. I was the spy! Was it the {location}?",
    "All your efforts were in vain. The spy was me and the location is the {location}!",
    "You were all so close, yet so far. I was the spy all along! Was it the {location}?",
    "The spy was me! I think it's the {location}.",
    "I’ve been pulling the strings from behind the scenes. I am the spy! Was it the {location}?",
)

SPY_GUESS_RIGHT_RESPONSE = (
    "You got us! That’s the right location!",
    "You got us! That’s right!",
    "We should have known it was you! You got it right!",
    "You got us! That’s the correct location!",
    "Ah, you got us! That’s the right location!",
)

SPY_GUESS_WRONG_RESPONSE = (
    "Nope! It was the {location}.",
    "No, it was the {location}.",
    "Close, but no! It was the {location}.",
    "Nope! It was the {location}. We win!",
)

ACCUSATION = (
    "I think it's {spy}. Are you the spy?",
    "I accuse {spy} of being the spy. Are you the spy?",
    "I suspect {spy} of being the spy. Is it you?",
    "I have a feeling it's {spy}. Are you the spy?",
    "I think {spy} is the spy. Are you the spy?",
)

SPY_INDICTED_RESPONSE = (
    "Ah, you got me! I am the spy.",
    "You got me! I am the spy.",
    "You caught me! I am the spy.",
    "Guilty as charged! I am the spy.",
    "Yep, it was me!",
)

NON_SPY_INDICTED_RESPONSE = (
    "No, I am not the spy.",
    "You’re wrong! I am not the spy.",
    "I am not the spy.",
    "You’re mistaken! I am not the spy.",
    "Nope, not the spy.",
)

NO_ONE_INDICTED_RESPONSE = (
    "Game over! Who was the spy?",
    "Game's over! Who's the spy?",
    "Looks like no one was was indicted. Who was the spy?",
)

SPY_REVEAL = (
    "Muah ha ha! I was the spy all along!",
    "Jokes on you all! I was the spy!",
    "You never suspected me, did you? I was right under your noses!",
    "Muah ha ha! Y'all played right into my hands! I was the spy!",
    "All your efforts were in vain. I was the spy all along!",
    "You were all so close, yet so far. I was the spy all along!",
    "I’ve been pulling the strings from behind the scenes. I am the spy!",
)
