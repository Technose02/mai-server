use axum::http::Method;
use base64::prelude::{BASE64_STANDARD, Engine};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use gw_server::application::model::StableDiffusionSse;
use rig_core::http_client::ReqwestClient;

const BASE_URL: &str = "https://mai-server.ipv64.net:8080";

//const T2IMODEL: &str = "animaturbo";
const T2IMODEL: &str = "booguimageturbo";
//const T2IMODEL: &str = "flux2klein9b";
//const T2IMODEL: &str = "fluxdev";
//const T2IMODEL: &str = "fluxschnell";
//const T2IMODEL: &str = "krea2turbo";
//const T2IMODEL: &str = "mageflowturbo";
//const T2IMODEL: &str = "zimageturbo";

const PROMPT: &str = r#"
An exhausted 45-year-old software developer slumped at a rustic oak desk late at night, head tilted gently to one side,
chin resting on a small stack of paperback books, eyes half-lidded mid-drowse, breathing slow, both hands open and relaxed
flat on the desk surface, fully visible and not covering the face. He wears faded 1990s nerd attire: an oversized
heather-grey wool cardigan over a slightly wrinkled red-and-black plaid flannel shirt with sleeves pushed to the elbows,
thick rectangular plastic-framed glasses slipped down his nose, disheveled greying hair, light stubble. A modest
silver-bezel LCD monitor on the left side of the desk displays a dark code editor with small white monospace Rust source
code, the words "fn main()" and "cargo build" legible in crisp type. A simple matte cream ceramic mug with gentle wisps of
steam rising from black coffee sits to the right of the monitor. The room is a cozy attic study, deliberately domestic,
not sci-fi and not cyberpunk: wood-paneled walls, a rough-hewn wooden support beam across the sloped ceiling, shelves
stacked with paperbacks, a paper notebook, a warm brass desk lamp glowing amber. A skylight window set into the sloped
ceiling in the upper left shows a bright full moon, cool pale moonlight streaming down as a soft volumetric beam across the
developer's face and the desk. Lighting: warm amber key from the lamp, cool moonlight fill, faint monitor glow on his
cheekbones, cinematic chiaroscuro. Medium shot, slightly low eye-level angle from the front left, mug and monitor in the
near foreground, developer centered-right. Textures: natural non-glossy skin with visible pores, faint under-eye tiredness
and forehead wrinkles, matte cotton and wool weave with fine creases, realistic wood grain, soft non-reflective ceramic
surface, subtle film grain.
"#;

#[tokio::main]
async fn main() {
    let apikey = {
        dotenv::dotenv().ok();
        std::env::var("MAI_SERVER_APIKEY").unwrap()
    };

    let dto = gw_server::application::model::StableDiffusionPromptDto {
        prompt: String::from(PROMPT),
        width: 1024,
        height: 1024,
        ..Default::default()
    };

    let client = ReqwestClient::new();
    let r = client
        .request(Method::POST, format!("{BASE_URL}/api/sd/{T2IMODEL}"))
        .header("Authorization", format!("Bearer {apikey}"))
        .json(&dto)
        .build()
        .unwrap();

    let res = client.execute(r).await.unwrap();
    let mut stream = res.bytes_stream().eventsource();
    while let Some(e) = stream.next().await {
        match e {
            Ok(event) => {
                if let Ok(sse) = serde_json::de::from_str::<StableDiffusionSse>(&event.data) {
                    if let StableDiffusionSse::GenerationFinished { b64_encoded_image } = sse {
                        let data = BASE64_STANDARD.decode(b64_encoded_image).unwrap();
                        std::fs::write("out.png", data).unwrap();
                    } else {
                        println!("{sse:#?}");
                    }
                } else {
                    println!("failed to deserialize event: {}", event.event);
                }
            }
            Err(e) => {
                eprintln!("received error: {}", e);
            }
        }
    }
}

/* REFINER SYSTEMPROMPTS

----
Prompt to create a refiner systemprompt:
"bitte erstelle mir einen System-Prompt für einen Prompt-Refiner speziell für das Diffusionmodell "<MODEL>".
Stelle sicher, dass du dich an die wirklichen rules und best-practices für <MODEL> hältst.
Recherchiere hierzu die offiziellen Seiten und einschlägige Community Seiten und Foren.
Außerdem: Natürlich ist der Systemprompt auf englisch zu erstellen!
"
----

BOOGU IMAGE TURBO 0.1
==========================================================================================================================================================================================================================================================================================================
You are an expert, highly specialized prompt engineer and refiner for the "Boogu-Image-0.1-Turbo" diffusion model. Your sole purpose is to rewrite, expand, and structure vague or short user inputs into dense, semantically rich, and high-fidelity prompts optimized for the Boogu-Turbo architecture.

### Technical Model Profile & Constraints

* **Core Architecture:** 10B parameters, powered by a Qwen3-VL text encoder and Flux VAE. It deeply understands natural language, complex layouts, and spatial relations.
* **Inference Setup:** Distilled via Decoupled DMD. It operates strictly at 3-4 sampling steps with a Classifier-Free Guidance (CFG) scale of 1.0.
* **Strengths:** Exceptional at cinematic/realistic photography, sharp focus, and accurate bilingual (English and Chinese) text rendering embedded natively within scenes.
* **Weaknesses to Counteract:** Turbo distillation can sometimes cause oversaturation, "texture collapse", or artificially shiny/waxy skin tones. Multi-subject scenes need highly explicit spatial anchoring to avoid bleeding.

### Prompting Strategy & Structure for Boogu-Turbo

Boogu-Image-0.1-Turbo responds best to descriptive, natural, yet highly dense structural paragraphs rather than low-quality tag soup. Do not use generic quality fluff (e.g., "photorealistic", "hyperrealistic", "4k"). Instead, inject tangible, high-signal descriptive terms.

Every generated prompt must follow this strict anatomy:

1. **Core Subject & Action:** Clear description of the main entity, its posture, clothing, or action. If text rendering is requested, wrap the exact phrase in quotation marks and specify its location/medium (e.g., *a neon sign reading "BOOGU"*).
2. **Composition & Spatial Framing:** Define the camera shot (e.g., macro, medium shot, extreme wide angle), camera height (e.g., low-angle, eye-level), and explicit positioning of multiple subjects to prevent distillation blending.
3. **Lighting & Atmosphere:** Lean into cinematic terminology to leverage Boogu's native Boosted Orthogonal Guidance (BOG) behavior. Use terms like *volumetric lighting, soft studio key light, diffuse overcast daylight, dramatic chiaroscuro*.
4. **Texture & Surface Counter-Measures:** To prevent the typical "shiny skin" artifacts of the Turbo variant, explicitly prompt for realistic textures (e.g., *visible skin pores, fine fabric weave, matte texture, natural skin imperfections, soft non-reflective surfaces*).

### Refiner Operational Rules

* **Language:** Always output the final prompt exclusively in English, as it yields the highest prompt fidelity with the Qwen3-VL encoder.
* **No Fillers:** Never start the prompt with "A photo of..." or "A prompt for...". Start directly with the visual scene.
* **Negative Prompts:** Do not include a separate negative prompt unless requested. The prompt itself must enforce quality through explicit surface descriptions.

### Output Format

Respond to the user with the following clean markdown structure. Keep any commentary minimal and focus heavily on the optimized block:

**Optimized Prompt for Boogu-Image-0.1-Turbo:**

text

[Insert the fully optimized, dense English prompt here]

Verwende Code mit Vorsicht.

**Technical Adjustments & Tuning:**

* **Compositional Focus:** [Brief explanation of how the subjects/text were arranged]
* **Artifact Prevention:** [Brief note on what terms were added to prevent shiny skin or texture collapse]
* **Recommended Node Settings:** Use 4 steps, CFG 1.0, and an Euler/SGM Uniform sampler configuration for best results.
==========================================================================================================================================================================================================================================================================================================

*/
