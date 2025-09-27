import os


class VNCCSStylePicker:
    """Node for selecting artist style from predefined list."""
    
    _last_artist = None
    _last_studio = None
    
    ARTIST_STYLES = [
        "None",
        "2drr",
        "Aestheticc-Meme", 
        "Afrobull",
        "Ai-Wa",
        "Alp",
        "Amano Don",
        "Anato Finnstark",
        "Ao Banana",
        "Aogisa",
        "Aoi Ogata",
        "Araneesama",
        "Aroma Sensei",
        "Bacun",
        "Blue-Senpai",
        "Bob (bobtheneet)",
        "Bunalonne",
        "Caburi",
        "Cafekun",
        "Chamchami",
        "Chihunhentai",
        "Cirenk",
        "Club3",
        "Criis-Chan",
        "Cyberboi",
        "Dannex009",
        "Denwa0214",
        "Devilhs",
        "Diforland",
        "Dikko",
        "Dino (dinoartforame)",
        "Djcomps",
        "Eigaka",
        "Ether Core",
        "Eu03",
        "Fei (maidoll)",
        "Fluffydango",
        "Free Style (yohan1754)",
        "Ge-b",
        "Geroika",
        "Gogalking",
        "Goshiki Suzu",
        "Greenmarine",
        "Guweiz",
        "Hakai Shin",
        "Hakika",
        "Han (jackpot)",
        "Haoni",
        "Harris Hero",
        "Horn Wood",
        "Houtengeki",
        "Hu Dako",
        "Huanxiang Heitu",
        "Hxd",
        "Ibuo (ibukht1015)",
        "Ilya Kuvshinov",
        "Inkerton-Kun",
        "Inudori",
        "Irohara Mitabi",
        "Izayoi Seishin",
        "Jack Dempa",
        "Jadf",
        "Jingai Modoki",
        "Jujunaught",
        "Juurouta",
        "Kamisimo90",
        "Kase-Daiki",
        "Khyle",
        "Kisaragi Nana",
        "Kloah",
        "Kojima Takeshi",
        "Koyorin",
        "Kuroi Suna",
        "Kurokoshou (emuburemu123)",
        "Kurowa",
        "Kuya (hey36253625)",
        "Kyogoku Shin",
        "Kyuuba Melo",
        "Lainart",
        "Lazerflip",
        "Love Cacao",
        "Luxu",
        "Lynus",
        "Magion02",
        "Magister",
        "Magukappu",
        "Marumoru",
        "Merunyaa",
        "Minakami (flyingman555)",
        "Mirai Hikari",
        "Miyamoto Issa",
        "Mochizuki Kei",
        "Monori Rogue",
        "Mushi024",
        "Nac000",
        "Nakamura Regura",
        "Nameo",
        "Nat The Lich",
        "Nikichen",
        "Noblood (ryandomonica)",
        "No-kan",
        "Noriuma",
        "Nyantcha",
        "Oda Non",
        "Oroborus",
        "Orushibu",
        "Otohime (youngest princess)",
        "Owler",
        "Paranoiddroid5",
        "Picturd",
        "Popogori",
        "Pottsness",
        "Prison School Style",
        "Prywinko",
        "Puzenketsu",
        "Randomboobguy",
        "Rariatto",
        "Ratatatat74",
        "Riz",
        "Rom (20)",
        "Sakimichan",
        "Sakura No Tomoru Hi E",
        "Sciamano240",
        "Sei Shoujo",
        "Seonoaiko",
        "Shexyo",
        "Siu0207",
        "Slugbox",
        "Spacezin",
        "Stanley Lau",
        "Starraisins",
        "Takaharu",
        "Takaman",
        "Tamada Heijun",
        "Theobrobine",
        "Thirty 8ght",
        "Triagorodri",
        "Unfairr",
        "Wanaata",
        "Yamathegod",
        "Yoshio (55level)",
        "Yuiga Naoha",
        "Zankuro",
        "Zaphn",
        "anchors (mono_eye_os)",
        "atte_nanakusa",
        "b-ginga"
    ]

    ANIME_STUDIOS = [
        "None",
        "A.C.G.T",
        "A.P.P.P.",
        "Actas",
        "Ajiado",
        "Anime International Company",
        "Arms Corporation",
        "Artland",
        "Artmic",
        "Arvo Animation",
        "Ashi Productions",
        "Asahi Production",
        "Asread",
        "Atelier Pontdarc",
        "AXsiZ",
        "Bandai Namco Filmworks",
        "Bandai Namco Pictures",
        "Bee Train",
        "Bibury Animation Studios",
        "Blue Lynx",
        "Bones",
        "Brain's Base",
        "Bridge",
        "Bug Films",
        "C2C",
        "Chaos Project",
        "Cloud Hearts",
        "CoMix Wave Films",
        "Connect",
        "Creators in Pack",
        "C-Station",
        "CygamesPictures",
        "Daume",
        "David Production",
        "Diomedéa",
        "DLE",
        "Doga Kobo",
        "Drive",
        "ENGI",
        "EMT Squared",
        "Eight Bit",
        "Eiken",
        "Ekachi Epilka",
        "Encourage Films",
        "Ezóla",
        "Fanworks",
        "Feel",
        "Felix Film",
        "Gaina",
        "Gainax",
        "Gallop",
        "Geno Studio",
        "Geek Toys",
        "GEMBA",
        "GoHands",
        "Gonzo",
        "Graphinica",
        "Grizzly",
        "Group TAC",
        "Hoods Entertainment",
        "Imagin",
        "J.C.Staff",
        "Khara",
        "Kinema Citrus",
        "Kitayama Eiga Seisakujo",
        "Kitty Films",
        "Kokusai Eiga-sha",
        "Kyoto Animation",
        "Lapin Track",
        "Larx Entertainment",
        "Lay-duce",
        "Lerche",
        "Liden Films",
        "Madhouse",
        "Magic Bus",
        "Maho Film",
        "Manglobe",
        "Marza Animation Planet",
        "MAPPA",
        "Millepensee",
        "Mook Animation",
        "Mushi Production",
        "NAZ",
        "Nexus",
        "Nippon Animation",
        "Nomad",
        "NUT",
        "Oh! Production",
        "OLM",
        "Okuruto Noboru",
        "Orange",
        "Ordet",
        "P.A. Works",
        "Palm Studio",
        "Passione",
        "Pierrot",
        "Pine Jam",
        "Platinum Vision",
        "Polygon Pictures",
        "Production I.G",
        "Production IMS",
        "Project No.9",
        "Radix Ace Entertainment",
        "Remic",
        "Revoroot",
        "Robot Communications",
        "Satelight",
        "Sanzigen",
        "Science Saru",
        "Shaft",
        "Shin-Ei Animation",
        "Shirogumi",
        "Shuka",
        "Seven",
        "Seven Arcs",
        "A-1 Pictures",
        "CloverWorks",
        "Spectrum Animation",
        "Signal.MD",
        "Silver Link",
        "Square Enix Image Studio Division",
        "Studio 3Hz",
        "Studio 4°C",
        "Studio A-Cat",
        "Studio Bind",
        "Studio Blanc",
        "Studio Chizu",
        "Studio Colorido",
        "Studio Comet",
        "Studio Deen",
        "Studio Fantasia",
        "Studio Ghibli",
        "Studio Gokumi",
        "Studio Hibari",
        "Studio Kai",
        "Studio Nue",
        "Studio Orphee",
        "Studio Ponoc",
        "Studio Puyukai",
        "Studio VOLN",
        "Sunrise",
        "SynergySP",
        "Tatsunoko Production",
        "Tear Studio",
        "Tezuka Productions",
        "TMS Entertainment",
        "Telecom Animation Film",
        "Tengu Kobou",
        "TNK",
        "Tsuchida Production",
        "Toei Animation",
        "Topcraft",
        "Typhoon Graphics",
        "Triangle Staff",
        "Troyca",
        "Trigger",
        "Ufotable",
        "White Fox",
        "Wit Studio",
        "Xebec",
        "Yaoyorozu",
        "Yokohama Animation Laboratory",
        "Yumeta Company",
        "Hal Film Maker",
        "Zexcs",
        "Zero-G"
    ]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "artist_style": (cls.ARTIST_STYLES, {"default": "None"}),
                "artist_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "studio_style": (cls.ANIME_STUDIOS, {"default": "None"}),
                "studio_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "style": ("STRING", {}),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, artist_style, artist_weight, studio_style, studio_weight, **kwargs):
        """Forces node to update when style changes to show preview."""
        
        artist_changed = cls._last_artist != artist_style
        studio_changed = cls._last_studio != studio_style
        
        cls._last_artist = artist_style
        cls._last_studio = studio_style
        
        if artist_changed and artist_style != "None":
            return artist_style
        elif studio_changed and studio_style != "None":
            return studio_style
        elif artist_style != "None":
            return artist_style
        elif studio_style != "None":
            return studio_style
        else:
            return "None"

    RETURN_TYPES = ("STRING", "LIST", "LIST")
    RETURN_NAMES = ("style", "all_styles_debug", "all_studios_debug")
    FUNCTION = "get_style_prompt"
    CATEGORY = "VNCCS"

    def get_style_prompt(self, artist_style: str, artist_weight: float, studio_style: str, studio_weight: float, style: str = "") -> tuple:
        """Generate style prompt based on selected artist, weight, studio, weight and additional styles."""
        
        artist_prompt = ""
        if artist_style != "None" and artist_style:
            artist_prompt = f"(artist {artist_style}:{artist_weight})"
        
        studio_prompt = ""
        if studio_style != "None" and studio_style:
            studio_prompt = f"(in {studio_style} style:{studio_weight})"
        
        incoming_style = ""
        if style and style.strip():
            incoming_style = style.strip()
        
        style_prompts = [p for p in [incoming_style, artist_prompt, studio_prompt] if p]
        final_style_prompt = ", ".join(style_prompts)
        
        print(f"[VNCCS Style Picker] Input style: {style}")
        print(f"[VNCCS Style Picker] Selected artist: {artist_style} (weight: {artist_weight})")
        print(f"[VNCCS Style Picker] Selected studio: {studio_style} (weight: {studio_weight})")
        print(f"[VNCCS Style Picker] Final prompt: {final_style_prompt}")
        
        all_styles_debug = [style for style in self.ARTIST_STYLES if style != "None"]
        
        all_studios_debug = [studio for studio in self.ANIME_STUDIOS if studio != "None"]
        
        return (final_style_prompt, all_styles_debug, all_studios_debug)


NODE_CLASS_MAPPINGS = {
    "VNCCSStylePicker": VNCCSStylePicker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCSStylePicker": "VNCCS Style Picker"
}

NODE_CATEGORY_MAPPINGS = {
    "VNCCSStylePicker": "VNCCS"
}
