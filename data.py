from enum import Enum

# This file contains useful data for agents to use ###############################


# All possible locations in the game
class Location(Enum):
    AIRPLANE = "Airplane"
    BANK = "Bank"
    BEACH = "Beach"
    BROADWAY_THEATER = "Broadway Theater"
    CASINO = "Casino"
    CATHEDRAL = "Cathedral"
    CIRCUS_TENT = "Circus Tent"
    CORPORATE_PARTY = "Corporate Party"
    CRUSADER_ARMY = "Crusader Army"
    DAY_SPA = "Day Spa"
    EMBASSY = "Embassy"
    HOSPITAL = "Hospital"
    HOTEL = "Hotel"
    MILITARY_BASE = "Military Base"
    MOVIE_STUDIO = "Movie Studio"
    OCEAN_LINER = "Ocean Liner"
    PASSENGER_TRAIN = "Passenger Train"
    PIRATE_SHIP = "Pirate Ship"
    POLAR_STATION = "Polar Station"
    POLICE_STATION = "Police Station"
    RESTAURANT = "Restaurant"
    SCHOOL = "School"
    SERVICE_STATION = "Service Station"
    SPACE_STATION = "Space Station"
    SUBMARINE = "Submarine"
    SUPERMARKET = "Supermarket"
    UNIVERSITY = "University"


# A dictionary that can optionally be used by agents to redact LLM output based on the location
# fmt: off
redaction_dict = {
    Location.AIRPLANE: ["airplane", "plane", "aircraft", "jet", "flight attendant", "flight", "pilot", "cockpit", "cabin", "turbulence", "boarding", "takeoff", "landing"],
    Location.BANK: ["bank", "money", "vault", "teller", "robbery", "robber", "heist", "cash", "rob", "atm", "loan", "account", "interest", "deposit", "safe"],
    Location.BEACH: ["beach", "sand", "ocean", "surf", "waves", "sun", "shell", "towel", "umbrella", "lifeguard"],
    Location.BROADWAY_THEATER: ["broadway theater", "broadway", "theater", "stage", "musical", "curtain", "play", "actor", "audience", "script"],
    Location.CASINO: ["casino", "gamble", "chips", "cards", "slot", "blackjack", "dealer", "roulette", "poker", "bet", "baccarat", "craps", "high roller"],
    Location.CATHEDRAL: ["cathedral", "church", "priest", "altar", "choir", "holy", "religion", "saint", "mass", "bishop", "nun", "confession"],
    Location.CIRCUS_TENT: ["circus tent", "circus", "tent", "clown", "ringmaster", "elephant", "juggler", "trapeze", "lion", "acrobat"],
    Location.CORPORATE_PARTY: ["corporate party", "corporate", "party", "boss", "employee", "office", "colleagues"],
    Location.CRUSADER_ARMY: ["crusader army", "crusader", "crusade", "knight", "sword", "war", "shield", "battle", "siege"],
    Location.DAY_SPA: ["day spa", "spa", "massage", "facial", "sauna", "pamper", "steam room", "treatment", "manicure"],
    Location.EMBASSY: ["embassy", "diplomat", "ambassador", "visa", "passport", "consulate", "foreign"],
    Location.HOSPITAL: ["hospital", "doctor", "nurse", "patient", "surgery", "medic", "stretcher", "ward", "emergency", "diagnosis", "prescription"],
    Location.HOTEL: ["hotel", "concierge", "lobby", "room service", "suite", "reception"],
    Location.MILITARY_BASE: ["military base", "military", "base", "tank", "soldier", "barracks", "general", "mission", "drill"],
    Location.MOVIE_STUDIO: ["movie studio", "movie", "studio", "film", "director", "set", "producer", "actor", "props", "script", "camera"],
    Location.OCEAN_LINER: ["ocean liner", "liner", "cruise", "deck", "captain", "port", "voyage"],
    Location.PASSENGER_TRAIN: ["passenger train", "train", "conductor", "rail", "track", "station", "platform", "ticket", "carriage"],
    Location.PIRATE_SHIP: ["pirate ship", "pirate", "treasure", "parrot", "plank", "cannon", "sail", "buccaneer", "map", "cutlass", "anchor", "mutiny"],
    Location.POLAR_STATION: ["polar station", "polar", "snow", "ice", "cold", "frozen", "research", "blizzard", "penguin"],
    Location.POLICE_STATION: ["police station", "police", "officer", "arrest", "criminal", "badge", "detective", "handcuff", "cell", "investigation", "siren", "warrant"],
    Location.RESTAURANT: ["restaurant", "waiter", "chef", "menu", "dine", "meal", "table", "food", "drink", "reservation", "kitchen", "bill"],
    Location.SCHOOL: ["school", "teacher", "principal", "teach", "subject", "homework", "classroom", "lesson"],
    Location.SERVICE_STATION: ["service station", "station", "gas", "mechanic", "repair", "fuel", "oil", "car", "tires", "service", "tune-up", "garage", "brakes"],
    Location.SPACE_STATION: ["space station", "space", "station", "astronaut", "orbit", "zero gravity", "gravity", "shuttle", "capsule", "moon", "nasa", "spacex", "module", "experiment", "commander"],
    Location.SUBMARINE: ["submarine", "underwater", "torpedo", "sonar", "deep", "periscope", "dive", "hatch", "crew"],
    Location.SUPERMARKET: ["supermarket", "store", "cashier", "cart", "groceries", "checkout", "barcode", "aisle", "product", "bag"],
    Location.UNIVERSITY: ["university", "student", "professor", "lecture", "campus", "degree", "exam", "college", "graduate", "teach", "subject", "faculty", "research", "lecture hall"],
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
