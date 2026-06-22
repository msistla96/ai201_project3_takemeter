# ai201_project3_takemeter

## Dataset specifications

### Community

The community chosen is the Subreddit r/anime. This has been present since 2008 and provides the most diverse ranges of posts and comments, which allows me to collect a diverse set of data to provide a classifier that can be as accurate as possible.

### Label Taxonomy

#### Discussion

A post that invites other people to share, debate, or react to a topic. 

Example:

    "When I say that everyone should watch it, I think the anime should be revolutionary in its stance about something. For instance, it should probably go against a commonly preconceived notion that a lot of people share, and it should express what it is in a great way. It can also just be absolute cinema and have an amazing plot such as AOT. It can also be an anime that changed your life, that you think a lot of people could benefit from watching. Personally my choice is either AOT or Your Lie in April, but I'm really curious to see what you guys think. Thanks!"

    "For some time now I’ve been wondering why do so many people hate the love triangle trope. If theres somebody who doesn’t like it can you please explain. From my perspective I believe it bring fun and all the “best girl” opinions and stuff. And btw i want to watch a new romance anime, any recommendations"


#### Review
A post that gives a more direct personal judgment of the topic with reasons and usually a clear overall opinion. Tends to be longer than a discussion.

Example: 

    "I have been holding back from watching this because of the way people talking about it, the memes and the comments criticising the anime. I avoided this anime for so long and finally went to see what the bad thing people has been talking about.  It’s not that bad as it seems like other people have been talking about. I thought it would be just full up fan service trash anime. Binged the two season this last week and thought it was great. The side characters are very great most of them are interesting. For the MC, i keep seeing how bad he was and how much people hate this guy. I thought this guy was straight up evil like people been saying but the controversy parts are like only just little bit scenes of classical anime fan service. The plot is interesting and the worldbuilding is just really good.  The Controversial parts are also pretty tame for an anime, There are many anime and anime characters who did the same thing as he did. The writing is quite old fashioned so it makes sense why the writer doesn’t hold back. With the time period, i can see why there are cousin marriages and marriages at a young age. Im a big fan of Game of thrones and ASOIAF universe so i don’t understand why this gets so much criticism compared to George books. I only assume that younger audiences or someone who is not familiar with medieval era and cannot handle mature themes that dislike this anime.  Not only that, but it seems like the only bad writing people have towards this show is only the MC and everything else is fine and good.  TLDR this anime is okay, not that bad as people say, the controversy is quite exaggerated, definitely not for everyone that can’t handle this kind of theme."

    "This is my first time visiting this community so please be gentle with me. I have never written a review of an anime, despite having always watched pretty much every new thing on crunchyroll with my wife, for years. I didn't know what I was about to find when I started this but I have to say this is probably one of the best anime I have ever experienced. I say experienced because I didn't just watch it. Growing up in a family of 6 kids we loved the power rangers and all the big spectacle action shows in the mornings. We would run around in the fields and woods pretending to be our favorite rangers fighting the monsters and bad guys. My older brother and I were the oldest boys and he was our leader. He passed unexpectedly 3 years ago and we have all processed it. This show however reached down into my soul and pulled up a lot that I had just forgotten because life moves fast and time buries a lot. My wife didn't get why I was constantly on the edge of my seat with tears in my eyes. It was because I saw my brothers and sisters up on that screen. I am sitting hear crying just writing this. It wasn't just the amazing animation and the stellar voice acting which both get 10/10 from me. it was how every single character was relatable and perfectly fit into their narrative. Everything about this one felt crafted just for me in a way. Maybe you all won't feel the same way about it, and that is perfectly ok. I just wanted to voice what was inside me and this anime made me feel I HAD to write this or i would just burst. Thanks and hope you all find your Tojima in something out there."



### Hard edge cases

When a discussion includes a review as the author tries to tell what they felt and the tone implies it's open to discussion.

### Data collection plan

Reddit archives through  `old.reddit` as it's free and easy to extract examples with less noise (There are a lot of videos, images, irrelevant text) compared to scrapping from Arctic Shift.
Since it's a community focused on discussions and reviews, there are sufficient examples for each kind of label that it can be a balanced set. 

### Data labelling plan

I initially started off with 6 labels: discussion, reaction, analysis, shitpots, opinion and review but I encountered the following issues specific for the r/anime community:

1. The community had a lot of outlier posts that did not fit neatly into any of the categories. It was difficult to scrap data as a result as scrapping would remove more than 50% of data. I tested multiple ways to filter the right posts but it was time-consuming and yielded inaccurate results.
2. Pure Analysis-only posts had 10-20 solid examples, which wasn't enough for balancing the dataset. A lot of analysis posts were blurred with discussion and reviews, which was more popular and easy to find.
3. Opinions and reactions weren't always obvious. Many of them were hidden behind pictures, videos and other links.

I then decided on the following 2 labels:

Discussion - 100 examples

Review - 100 examples

In the process of labelling, I struggled with some of the following examples:

    "So I watched The Garden of Words today. It's only a 45 minute movie, but it is so beautiful. The way the rain and the environment are depicted makes you truly feel it, watching the movie feels just like reading a piece of poetry. I really liked the characters and the story as well..."
    True Label: Discussion
    Reason: The post has all the markings of a review, but the informal language focusing on feelings made me want to reply to the post. 

    "So, I have finished watching the first season of Oshi no Ko. Although it has 11 episodes, the first episode was the length of four... Before starting the series, I had seen some pictures and clips where everything looked very cute, but once I started watching, I realized how much darkness     was hidden beneath that cuteness. It really highlights the negative side of the entertainment industry..."
    True Label: Discussion
    Reason: The post has all the markings of a review except for a final recommendation or verdict, and this opens it for discussion.


    "TL;DR: I'm enjoying winter 2026. So, as the Winter 2026 season is approaching the halfway point for several shows, I wanted to share some thoughts on my currently airing list..." followed by numbered per-show takes.
    True Label: Review
    Reason: The post explictly talks about sharing feelings, but dives into details, evaluations etc. The length reaches closer to a Review than a Discussion, helping me reflect on this as someone who has not seen all the shows.


## Fine tuning pipeline

Model: `distilbert-base-uncased`

Platform: `Google Colab`

Hyperparameters: 

Epochs: 5

Learning rate: 2e-5

Batch size for training: 16

Change to params:

Epoch

3 provided F1-scores under 0.6 for both labels, had 70% training accuracy but higher validation loss of around 60% and had 19/30 predictions labelled correctly in the test set, which was slightly above 50%(close to guessing). 
Larger epochs improved validation to less than 60%, provided stable training accuracies and had 24/30 predictions labelled correctly in the test set, increasing to 80% accuracy.


### Baseline with few shot prompt

Baseline used a few shot prompt with `openai/gpt-oss-120b` model for classification.

    SYSTEM_PROMPT = """
    You are classifying posts from r/anime.
    Assign each post to exactly one of the following categories.

    Discussion: A post that invites other people to share, debate, or react to a topic.

    Example: "I don't even want to name the anime's specifically because it isn't about one or a group of specific ones for me. I was never a hard anime over normal cinema person and still am not but I always was able to find some cool anime I enjoyed every once in a while. Over the last several years though there has been a shift and focus on anime production and quality. Both in quantity and quality. Some of these I cannot stop watching until 4am and some are even bringing even me to tears. To tears, I'm an early 30's guy. Even the extremely popular new TV shows and movies miss the marks in so many ways that I don't find in anime. They certainly don't evoke emotional responses from me like some of these anime's do. I watched another one recently and it is still hard in my mind days later. Even the soundtrack. It's not just my opinion either. Anime and crunchyroll have shown prolific growth and mainstream western adoption as of late. https://www.wsj.com/business/media/hollywoods-hottest-business-is-once-niche-anime-4d26ca5e
    Am I the only one taking notice of this?"

    Review: A post that gives a more direct personal judgment of the topic with reasons and usually a clear overall opinion.

    Example: "The reason I love Kaiji: Ultimate Survivor is because Kaiji is so human. He is deeply flawed, anxious, impulsive, and self destructive. But at the same time he is clearly a genius. If he was not an addict, if he was not constantly sabotaging himself, he would succeed in literally any aspect of life.

    His endurance is insane. His ability to calculate risk under pressure is unmatched. He keeps going when most people would have already broken. That is what makes him so compelling. He is not a power fantasy. He is potential trapped inside weakness. Watching him think, panic, adapt, and barely survive feels real in a way most anime never even attempt.

    In my opinion, it is the most underrated anime of all time. The voice acting is phenomenal. The soundtrack is insanely underrated and does so much heavy lifting for the tension. The intensity is constant and exhausting in the best way possible. And overall it is just completely unique.

    There is nothing else quite like it. It feels like if the narrator from Baki and JoJo’s Bizarre Adventure had a child, and that child turned out edgy, obsessed with Death Note, and really into gambling and we age grew up to be a componist for the Phoenix wright Games.

    Honestly, it is almost a masterpiece. I say almost only because the manga is not finished yet. And to everyone reading this, I genuinely recommend continuing with the manga. I almost never do that. I am usually anime only. But Kaiji is different. It is absolutely peak.

    I was skeptical at first because the next major arc is the mahjong arc, and I did not even understand mahjong. It did not matter. It was still absolutely fantastic. I later reread it after learning mahjong, and it somehow became even better. The real tragedy is that most anime only watchers never get introduced to the best villain in the entire series. You are missing out on something special. Do yourself a favor and give the manga a chance. You will not regret it."

    Respond with ONLY the label name.
    Do not explain your reasoning.

    Valid labels:
    Discussion
    Review
    """

After getting labels for each, the confusion matrix and accuracy metrics were calculated.

## Evaluation Report

## Fine-Tuned Model — Confusion Matrix (Test Set)

|  | **Predicted: Discussion** | **Predicted: Review** |
|---|---|---|
| **True: Discussion** | 10 ✅ | 5 ❌ |
| **True: Review** | 1 ❌ | 14 ✅ |

### Breakdown

| Metric | Discussion | Review |
|--------|------------|--------|
| True Positives | 10 | 14 |
| False Negatives | 5 | 1 |
| False Positives | 1 | 5 |


# Baseline vs Fine-Tuned

## Overall Accuracy

| Model | Accuracy | Parseable Responses |
|-------|----------|-------------------|
| Baseline | 0.767 (76.7%) | 30/30 |
| Fine-Tuned | 0.800 (80.0%) | — |
| **Δ Change** | **+0.033 (+3.3pp)** | |

---

## Per-Class Metrics

### Discussion

| Metric | Baseline | Fine-Tuned | Δ Change |
|--------|----------|------------|----------|
| Precision | 0.83 | 0.91 | +0.08 ✅ |
| Recall | 0.67 | 0.67 | 0.00 ➡️ |
| F1-Score | 0.74 | 0.77 | +0.03 ✅ |
| Support | 15 | 15 | — |


### Review

| Metric | Baseline | Fine-Tuned | Δ Change |
|--------|----------|------------|----------|
| Precision | 0.72 | 0.74 | +0.02 ✅ |
| Recall | 0.87 | 0.93 | +0.06 ✅ |
| F1-Score | 0.79 | 0.82 | +0.03 ✅ |
| Support | 15 | 15 | — |


## Prediction Results

## Sample Classifications

    The fine tuned classifier predicted 24/30 examples correctly. Below are some sample predictions:

    --- #1 ---
    Text:      What are everyone’s standout shows for the current season? So far mine are - Agents of the four Seasons, Daemons of the Shadow Realm, Which Hat Atelier, Nippon Sangoku, Marriage Toxin and Botan Kamiin...
    True:      Discussion
    Predicted: Discussion  (confidence: 0.73)
    Why: The opening sentence asks for a poll and shows Discussion. The post offers the author's own picks as a prompt for others to respond, with no verdicts, plot summaries, no pros/cons etc.
    
    --- #2 ---
    Text:      When I say that everyone should watch it, I think the anime should be revolutionary in its stance about something. For instance, it should probably go against a commonly preconceived notion that a lot...
    True:      Discussion
    Predicted: Discussion  (confidence: 0.77)
    Why: The post is abstract: it defines criteria for what would make an anime recommendable without evaluating any show in specific. It ends with "I'm really curious to see what you guys think,", which matches what a discussion would be.


    --- #3 ---
    Text:      I have a strange relationship with baseball. I come from a part of the world where the sport isn’t popular at all, and I myself am not very athletic. However, for some reason or another I am really fo...
    True:      Review
    Predicted: Review  (confidence: 0.51)
    Why: The post evaluates a specific anime (Catch Me At The Ballpark) through a personal baseball-outsider perspective. It builds toward an overall assessment of the show rather than asking a question.

    --- #4 ---
    Text:  This was surprisingly good.  I actually watched it a long time ago when it first aired in 2014 and it was better than I remembered.  It's a Studio Trigger anime but you'd be forvien for not knowing th...
    True:      Review
    Predicted: Review  (confidence: 0.73)
    Why: The post has text for rewatch comparison, studio attribution, and a breakdown of what the show is and why it works. The post ends with an overall positive assessment. 

    --- #5 ---
    Text:      I was rewatching Management of a Novice Alchemist (only the second rewatch, not really up there in my likes) &amp; the voice of the MC (Sarasa) tickled my brain. After a few episodes I closed my eyes ...
    True:      Discussion
    Predicted: Discussion  (confidence: 0.76)
    Why: The post is a story about recognising a voice actor during a rewatch and ends with "Anyone else had the experience where a VA for even a minor part tugged at you?". The show itself is barely evaluated ("not really up there in my likes" is the only opinion expressed). 



### Wrong predictions

The fine tuned classifier predicted 6/30 examples incorrectly. Below are some sample predictions:

    --- #1 ---
    Text:      I’m rewatching SAO S2 (yes, I know, it’s a guilty pleasure of mine), and during the GGO arc, there’s a scene where Kirito and Sinon have a discussion about their current situation and circumstances in...
    True:      Discussion
    Predicted: Review  (confidence: 0.81)
    Why: The list of shows with verdicts shows that it can be a review. However, there is a closing sentence "Please feel free to share your favorites!" that makes it a discussion. The model considers the majority text to be a Review.

    --- #2 ---
    Text:      I watched Suzume today, and I didn't think this movie would make me emotional. The setup was such that I thought in the end Souta would become fine, they would find a way to close all the doors, and t...
    True:      Discussion
    Predicted: Review  (confidence: 0.56)
    Why: The post contains a score ("9/10"), plot breakdown, and animation praise which can look like a review. However, the model failed to see that there was no stuctured breakdown and it had reactive feelings.

    --- #3 ---
    Text:      It’s definitely another one of my top favorites. It’s such a beautiful and relatable series about long term trauma and grief and the process of healing from that. I thought it would be your average te...
    True:      Discussion
    Predicted: Review  (confidence: 0.76)
    Why: The post contains a thematic analysis (trauma, grief, character arcs), criticism section ("My only nitpick"),and a conclusion. However, it is a reply in a thread; "It's definitely another one" implies agreement with a previous comment. The misclassification was due to lack of context.

    --- #4 ---
    Text:      Season 1 was good for me and i really liked how refreshing it was to see a good romance about people who love each other however.  Season 2 is nearly done and:  \- Side Characters genuinely don’t exis...
    True:      Discussion
    Predicted: Review  (confidence: 0.61)
    Why: The post has a structured, bullet-pointed critique ("Season 1 was good", "Season 2 is genuinely unwatchable"). The closing question arrives after four bullet points of criticism and is a single sentence. The model did not consider the last sentence.

    --- #5 ---
    Text:      Hello /r/anime!   With 2025 ending I thought it would be nice to look back at this past year of anime we had. This is the _sixth_ year ^(^and ^possibly ^last ^year) I'm doing this. [2020](https://redd...
    True:      Review
    Predicted: Discussion  (confidence: 0.56)
    Why: The post is comprehensive: scoring and ranking 90 anime. But it has a self-deprecating tone, provides award lists, and a closing prompt. The model got confused and leaned towards a Discussion.

    --- #6 ---
    Text:      Just got back from Crunchyroll Anime Nights Sneak Peek for Summer 2026! This one was not nearly as exciting as the Spring 2026 showing, but still had something I liked at least! Here are my thoughts: ...
    True:      Discussion
    Predicted: Review  (confidence: 0.77)
    Why; The post provides verdicts for a list of shows for Summer. Each show has a name, plot and a clear conclusion on whether to watch. The closing community question ("If you attended,what were your thoughts?") and opening "Just got back from Crunchyroll Anime Nights Sneak Peek for Summer 2026" makes it more open for a discussion, but it does not surface enough for the model, which considers the majority text to be a Review.


## Prediction Patterns

## Error Pattern Analysis — Fine-Tuned Model


### Error table

| # | True | Predicted | Confidence | Post Type |
|---|------|-----------|------------|-----------|
| E1 | Discussion | Review | 0.81 | Rewatch + scene list prompt |
| E2 | Discussion | Review | 0.56 | First-watch emotional reaction |
| E3 | Discussion | Review | 0.76 | Reply endorsing a show |
| E4 | Discussion | Review | 0.61 | Season comparison critique |
| E5 | Review | Discussion | 0.56 | Annual awards megapost |
| E6 | Discussion | Review | 0.77 | Live event report |


### Pattern 1: Skewed errors

5 of 6 errors are Discussion -> Review. The model has a systematic Review bias on ambiguous posts.

| Error type | Count | 
|-----------|-------|
| Discussion predicted as Review | 5 | 
| Review predicted as Discussion | 1 |

This aligns with the classification report: Discussion recall is 0.67 vs Review recall of 0.93. The model leans Review when uncertain, and uncertainty is mostly for Discussion posts rather than Review posts.

### Pattern 2: Placement, density, language and structure of Discussion signals matter

All 5 Discussion -> Review errors start with evaluative, impression-driven language before any community or discussion signals appear:

| Example | Opening sentence | Discussion Signal | Position of discussion signal |
|---------|---------------|-----------------|------------------------------|
| E1 (SAO) | "guilty pleasure", scene-by-scene praise | "Please feel free to share" | Final sentence |
| E2 (Suzume) | "I didn't think this would make me emotional", plot breakdown | None | Absent |
| E3 (Fruits Basket) | "one of my top favorites", theme analysis, nitpick | Implicit (reply thread) | Absent from text |
| E4 (GF²) | Season 1/2 comparison, bullet-point critique list | "just wondering if anyone else thinks similar?" | Final sentence |
| E6 (Crunchyroll) | Per-show spoiler verdicts, "not gonna watch further" | "If you attended, what were your thoughts?" | Final sentence |

The discussion signal is usually a single closing sentence. The model appears to use earlier tokens to predict the post as a Review, even though the end tokens show that it's a Discussion.

| Example | Closing Sentence | Confidence | Outcome |
|---------|-----------------|------------|---------|
| E1 | "Please feel free to share your favorites!" | 0.81 | ✗ Wrong |
| E4 | "just wondering if anyone else thinks similar?" | 0.61 | ✗ Wrong |
| E6 | "If you attended, what were your thoughts?" | 0.77 | ✗ Wrong |


### Pattern 3: Some Review -> Discussion errors are unique

E5 (Amewards annual post) is the only Review misclassified as Discussion,
and it is an outlier among Reviews in the dataset.

| Feature | Typical Review | E5 |
|---------|---------------|---------------|
| Audience address | Implicit ("the reader") | Explicit ("Hello /r/anime!") |
| Tone | Evaluative, authoritative | Conversational, self-deprecating |
| Closing | Summary verdict | Community prompt + social links |
| Length | Moderate | Very long (15 category sections) |
| Format | Prose paragraphs | Section headers + ranked lists |

The post is functionally a Review (evaluating 90 anime across a year) but invites discussion by the community. This maybe an out-of-distribution example due to rarity or dataset imbalance.


## Confidence Calibration

### Correctly Classified Examples

| # | True Label | Predicted | Confidence | Correct? |
|---|-----------|-----------|------------|----------|
| 1 | Discussion | Discussion | 0.73 | ✓ |
| 2 | Discussion | Discussion | 0.77 | ✓ |
| 3 | Review | Review | 0.51 | ✓ |
| 4 | Review | Review | 0.73 | ✓ |
| 5 | Discussion | Discussion | 0.76 | ✓ |


### Misclassified Examples

| # | True Label | Predicted | Confidence | Correct? |
|---|-----------|-----------|------------|----------|
| 1 | Discussion | Review | 0.81 | ✗ |
| 2 | Discussion | Review | 0.56 | ✗ |
| 3 | Discussion | Review | 0.76 | ✗ |
| 4 | Discussion | Review | 0.61 | ✗ |
| 5 | Review | Discussion | 0.56 | ✗ |
| 6 | Discussion | Review | 0.77 | ✗ |



### Findings

#### 1. High-confidence errors often correspond to overconfidence, truths to precision.
Examples #1 (0.81), #3 (0.76), and #6 (0.77) were all wrong with high confidence. For truthful labels, the high confidence leads to the right answer. These are troublesome because there is a genuine blurring of the boundary between Discussion and Review.
This is especially true for Discussion posts that are structurally ambiguous and surface Review-like patterns.

#### 2. Low-confidence errors and truths reflect model confusion accurately.
Examples #2 (0.56) and #5 (0.56) are near the decision boundary, the model wasn't sure how to classify. These are easier to flag and threshold as they are close to 0.50.

#### 3. Moderate confidence is hard to tune.
Example 4(0.61) lies between Findings 1 and 2. These are more likely to occur due to outliers, taxonomy limitations or dataset imbalance. 
Collecting more data for such examples for fine tuning or using a LLM-as-a-judge or human review for on-the fly prediction could help resolve this as thresholds wouldn't be reliable.


### Observations and Reflections

1. The Taxonomy for these labels are hard to fully define without testing as the text density, placement, structure and language are highly variable. 
2. Subclasses within a label are not obvious and can skew accuracy and metrics. One way to resolve ambuiguity within these subclasses is to provide more examples of what a Discussion looks like when it happens at the end specifically for `r/anime` could help with improving the recall for Discussion and hence confidence of the model for these examples. Further dividing these labels may help with adding stronger boundaries.
3. The model was supposed to be able to pickup on a post being a Discussion, but had issues with doing so. It did pretty well on Review posts, which I expected it to fail at.

## AI tool plan

### Label stress-testing
I used Claude to stress test my definitions and edge cases. I asked it to iterate 4-5 times until there was not much to iterate on.

### Annotation Assitance
I used Claude and Codex to help label some of the examples I collected in `discussion.txt` and `review.txt` so that I could see if my definitions of discussion and review were clear enough. Asking it to label all caused rate limiting issues and batching took too long.

### Failure Analysis
I supplied Claude with prediction examples and confusion matrix to see any relationship between them. Using the metrics it got, I added my own hypothesis for observations for why a metric is so i.e Recall for Discussion is higher than for Review.

## Specification Reflection

1. Using AI to stress test definitions helped me to iterate on my definitions and see that dicussion and review definitions vary across communities.
2. I iterated between Annotation Assistance and Label stress-testing steps, which helped me narrow down which labels had more examples and were better reflective of the community.
