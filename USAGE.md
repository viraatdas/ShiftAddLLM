## Usage
1. Quantize model
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 model/llama.py meta-llama/Llama-2-13b-hf --wbits 3 --groupsize 128 --acc --bcq --bcq_round 50 --load BCQ_ACC_Llama-2-13b-hf # bcq_round 20 works too, bigger - slower - maybe better
```

2. Generate
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 generate_text/generate_llama.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --load BCQ_LAT_Llama-3.1-8B-Instruct.pt --benchmark
```

3. Streamlit
On local:
```
ssh -L 8501:localhost:8501 ubuntu@150.136.66.218
```


On remote:
```
streamlit run chat/chat_app.py --server.port 8501 --server.address 0.0.0.0
```



## Results

1. BCQ_ACC_Llama-2-13b-hf 

```
ubuntu@150-136-43-160:~/llm-train-filesystem/ShiftAddLLM$ time CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 generate_text/generate_llama.py --model meta-llama/Llama-2-13b-hf --load BCQ_ACC_13b_HF --benchmark True
==== Benchmarking unquantized model ====
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████| 3/3 [00:00<00:00, 12.80it/s]
Generated Text (Unquantized Model):
Once upon a time, there was a little girl whose favorite color was pink.
Her mother and father loved her so much that they gave her everything she could possibly imagine. She had everything she could possibly want.
She had her own toys, her own computer, her own bed, her own everything. She didn't want for anything.
So, when she finally grew up and became a young woman, she became a little spoiled. She wanted everything that money could buy.
She loved jewelry and designer clothes. She loved to travel and live in the lap of luxury. She had everything she could possibly imagine.
One day she met a young man. He was dashing, handsome, and charming. And most of all, he had everything.
He had money and he loved her very much, and they married and lived in a huge mansion, with servants to wait on them hand and foot.
Soon, she became pregnant, and when the time came, she gave birth to a beautiful baby girl.
Now, the little girl had everything she could possibly imagine, just like her mother had everything she could possibly imagine.
The little girl didn't have any real toys to play with, though. She didn't have any friends to play with.
And she didn't have any real clothes to wear. She didn't have any real food to eat, she only
Generated Text (Unquantized Model):
In a galaxy far, far away, a legendary Star Wars comic book series gets the epic library treatment it deserves! Dark Horse Comics, the publishing house that brought fans the very best of Star Wars comics for over two decades, will release the first of three high-end hardcovers chronicling the adventures of everyone’s favorite mercenary in Star Wars: Dawn of the Jedi: Prisoners of Bogan, on sale January 22, 2014.
Star Wars: Dawn of the Jedi was one of the most ambitious Star Wars comics ever published, following the history of the Jedi Order back to its earliest days. With the Star Wars: The Clone Wars television show expanding the mythology of the Jedi back a thousand years, this is the ultimate collection of this fan-favorite series!
This first book of the series, Star Wars: Dawn of the Jedi: Prisoners of Bogan collects issues #1-5 of the series, and features a painted cover by Doug Wheatley, and a “behind-the-scenes” sketchbook section.
Star Wars: Dawn of the Jedi: Prisoners of Bogan goes on sale in bookstores and comic shops on January 22, 2014 for $34.99.

Generated Text (Unquantized Model):
The quick brown fox jumps over the lazy dog.
I love the coloring on this image. It is so vibrant and detailed. I am sure it will take me awhile to color this one. I just love the idea behind the image, the fox that is quick and smart, and the dog that is lazy and a bit slow. Just a great saying that will go great in our home.
As with all stamps, I started with the sentiment. In this case, it is an image that I am using for the background. I always do the image first, just a preference of mine. I used the Color Burst technique. I love the color that I got.
The image was colored with color pencils and colored pencils. I made a mask for the dog and just outlined it with the pencils. I also made a mask for the sentiment to make it easy to color.
After coloring, I cut it out with a Stitched Rectangle die.
I used the same sentiment from the image and colored it with the Color Burst technique. I adhered it to the card with some foam tape.
Next, I pulled out some of my favorite flowers and punched them out with a Flower Punch. The center has some glitter on it. I adhered the flowers with some Stick It, then foam taped them.
I
Generated Text (Unquantized Model):
In the middle of the night, she heard a sound and went to investigate. It was her husband standing by the front door in his pajamas. When she asked him what he was doing, he replied, "I was just waiting for you."
Sweetest Valentine's Day Story - SHARE
The Sweetest Valentine's Day Story I have heard all day.
A man bought his wife a CROCKPOT for their upcoming anniversary.
The instruction manual contained the following:
"For best results, put the entire contents of this pot in a blender."
"Sweetheart," said the wife to her husband.
"What would you like for your anniversary?"
"Oh, I don't need to shop, dear," he replied.
"This year, you can get me a CROCKPOT."
Sweetest Valentine's Day Story
"I want to buy my husband a special Christmas present."
"Why don't you buy him a CROCKPOT!"
"No, I'll probably get him a tie. He always says ties are uncomfortable. He says the thing he really wants is a CROCKPOT."
"Forget the tie, buy him the CROCKPOT!"
"I don't think so, I'd rather have a CROCKPOT
Generated Text (Unquantized Model):
The world was on the brink of change and only a young woman can save the world.
Genevieve is a young woman living in a post-apocalyptic world where humans and robots live in an uneasy truce. But when she finds a mysterious artifact in the woods that changes her life forever, she must embark on a quest to discover its origins and save the world from destruction. With her faithful robot companion by her side, Genevieve sets off on a journey filled with peril, danger, and unexpected twists and turns.
Will Genevieve be able to unlock the secrets of the artifact and save the world from impending doom? Or will the mysterious artifact destroy everything in its path? Find out in "The Artifact," a thrilling and action-packed adventure that will keep you on the edge of your seat.
The Artifact is an epic adventure film that follows the journey of a young woman named Genevieve as she discovers an ancient artifact that holds the power to change the world. With the help of her trusted companion, a robot named N.I.C.K., Genevieve embarks on a quest to uncover the secrets of the artifact and save humanity from an impending disaster.
However, their journey is not without obstacles, as they must battle against dangerous enemies and a corrupt government that seeks to
Generated Text (Unquantized Model):
It all started with a simple mistake: a mistake in judgment, an error in assumption.
I'd been working for a tech company for about a year after I graduated from college. I was assigned to an important project that had the potential to help our company reach its sales goals. I was tasked with creating marketing campaigns to promote the launch of a new product. It was an exciting project, one that would require me to think creatively and develop new skills. I was eager to get started.
I worked diligently on the project, staying late nights and on weekends, determined to make the launch a success. I researched the target audience and developed a marketing strategy that I believed would resonate with potential customers.
As the launch date approached, I was feeling confident in my work. But then, a few days before the launch, I made a mistake. I accidentally overwrote a key file with an earlier version, losing all of the work I had done.
I was panicked. I knew that I needed to fix the issue, but I had no idea how. I tried everything I could think of, but to no avail. I was up against the clock, and the launch date was rapidly approaching.
I felt helpless and overwhelmed. I had put so much time and energy into the project, and now, it was all gone. I felt like a failure. I
Tokens per second (Unquantized Model): 26.96 tokens/second
Tokens per second (Unquantized Model): 31.50 tokens/second
Tokens per second (Unquantized Model): 31.82 tokens/second
Tokens per second (Unquantized Model): 31.69 tokens/second
Tokens per second (Unquantized Model): 31.40 tokens/second
Tokens per second (Unquantized Model): 31.69 tokens/second
Avg TPS (Unquantized Model): 30.844200232066267
========================================

===== Running quantized model =====
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████| 3/3 [00:00<00:00,  9.84it/s]
/home/ubuntu/llm-train-filesystem/ShiftAddLLM/generate_text/generate_llama.py:82: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(model_checkpoint))
Generated Text (Quantized Model):
Once upon a time, a hero arose from humble beginnings to become the mightiest of them all. He was chosen from the heavens to come and save the world of Chaos, and he was the chosen one.
This was no lesser a feat than creating a whole new universe from nothing but your own mind and belief. When you are faced with a task as monumental as this, it is important to have a positive attitude and a positive mindset. When you are doing something as monumental as this, it is important to have a positive attitude and a positive mindset. This is because when you are doing something as monumental as this, it is important to have a positive attitude and a positive mindset.
When you are facing a task as monumental as this, it is important to have a positive attitude and a positive mindset. When you are doing something as monumental as this, it is important to have a positive attitude and a positive mindset. This is because when you are doing something as monumental as this, it is important to have a positive attitude and a positive mindset.
When you are facing a task as monumental as this, it is important to have a positive attitude and a positive mindset. When you are doing something as monumental as this, it is important to have a positive attitude and a positive mindset. This is because when you are doing something as monumental as this, it is
Generated Text (Quantized Model):
In a galaxy far, far away...
...a long, long time ago...
(Yes, I am a huge Star Wars geek! I have the original Star Wars trilogy on VHS, which are now collector's items. I also have the extended versions of Episodes 1-3 on DVD.)
Back to the topic at hand...I am not a huge fan of comic books, but I really like reading manga. I recently got into reading the Death Note series (in which a teenage boy named Light Yagami discovers that his death note has killed him).
I am now working on volume 2, in which Light and his partner, L, fight against the mafia in New York (well, sort of).
I love reading manga, and I have read quite a few. Here are a few of my favorites.
Death Note series (Japanese edition)
The Black Cat series
Nana (not the anime series)
Tokyo Babylon series (Japanese edition)
Yu-Gi-Oh! series (Japanese edition)
The following is my list of recommended reading order for those series. Keep in mind that some of these series are very complex. If you're not used to reading them, it might be hard to follow the story.
If you are reading the English version, start with volume 
Generated Text (Quantized Model):
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog, bounces off the tree and heads towards the lake.
The quick brown fox jumps over the lazy dog, bounces off the tree and heads towards the lake.
The quick brown fox jumps over the lazy dog, bounces off the tree and heads towards the lake.
The quick brown fox jumps over the lazy dog, bounces off the tree and heads towards the lake.
The quick brown fox jumps over the lazy dog, bounces off the tree and heads towards the lake.
The quick brown fox jumps over the lazy dog, bounces off the tree and heads towards the lake.
The quick brown fox jumps over the lazy dog, bounces off the tree and heads towards the lake.
The quick brown fox jumps over the lazy dog, bounces off the tree and heads towards the lake.
The quick brown fox jumps over the lazy dog, bounces off the tree and heads towards the lake.
The quick brown fox jumps over the lazy dog, bounces off the tree and heads towards the lake.
The quick brown fox jumps over the lazy dog, bounces off the tree and heads towards the lake
Generated Text (Quantized Model):
In the middle of the night, she heard a sound. She got up and checked on her son. He was still asleep. She went back to bed. Then she heard another sound. She got up and checked on her son again. He was still asleep. She went back to bed. Then she heard another sound. She got up and checked on her son again. He was still asleep. She went back to bed. Then she heard another sound. She got up and checked on her son again. He was still asleep. She went back to bed. Then she heard another sound. She got up and checked on her son again. He was still asleep. She went back to bed. Then she heard another sound. She got up and checked on her son again. He was still asleep. She went back to bed. Then she heard another sound. She got up and checked on her son again. He was still asleep. She went back to bed. Then she heard another sound. She got up and checked on her son again. He was still asleep. She went back to bed. Then she heard another sound. She got up and checked on her son again. He was still asleep. She went back to bed. Then she heard another sound. She got up and checked on her son again. He was still asleep. She went back to bed. Then she heard another sound. She got up and checked on her son
Generated Text (Quantized Model):
The world was on the brink of change. New nations would rise. The power of old countries would dwindle. The United States of America, Great Britain, France and Germany would grow from strength to strength. Russia would shrink and die. Japan would become the most powerful nation of the East. China would be a great power too. All these things would come to pass.

[Illustration: _They would build their homes on solid ground._]

[Illustration: _They would fight their battles with new weapons._]

[Illustration: _They would carry their wars across the seas._]

[Illustration: _They would become the masters of the sky._]

But the most wonderful of all would be the wonder that would come to pass in space--the wonder of the new men of the new worlds who would come to be masters of the universe.

CHAPTER IV

_The Titanic Adventure_

They were out of space now, the three men who had come so far across the vastness of the universe in the spaceship they had built.

At last they had reached it--the place where they were to begin their greatest adventure. It was called, simply enough, Outer Space.

And they had come to it in a spaceship they had made out of the steel they had found in the great mountains of rock.

Out
Generated Text (Quantized Model):
It all started with a simple mistake. When five-year-old Jamie and I moved from England to Scotland it was foggy and Jamie was so excited about our new life and couldn’t wait to get going.
When I told Jamie he had to go slowly down the stairs he said “I don’t have to” so I said “OK then.” He ran down the stairs and tripped, falling onto the bottom stairs and badly injuring his knee.
We took him to the hospital and he needed an operation, which we went to the next day. He then spent 2 weeks in hospital recovering.
That was the day that changed our lives forever. I didn’t know it at the time but my son was going to be forever changed and so was I.
When he was discharged from the hospital they gave us the news that Jamie would require regular check-ups at the hospital to ensure everything was ok and if there was any problems they would spot them early.
This was a shock to us both as I had never heard of this condition before. I was worried about how we would cope and also worried about what this would mean to Jamie.
We were then referred to the orthopedic department for children’s surgeries.
They saw him weekly and made sure he was in good health and also checked his legs for any problems.
At the hospital
Tokens per second (Quantized Model): 31.37 tokens/second
Tokens per second (Quantized Model): 31.81 tokens/second
Tokens per second (Quantized Model): 32.29 tokens/second
Tokens per second (Quantized Model): 32.26 tokens/second
Tokens per second (Quantized Model): 32.09 tokens/second
Tokens per second (Quantized Model): 31.91 tokens/second
Avg TPS (Quantized Model): 31.96 tokens/second
===================================

real    3m20.195s
user    2m11.265s
sys     1m17.537s

```



2. BCQ_ACC_Llama-3.1-8B-Instruct

```
ubuntu@150-136-66-218:~/llm-train-filesystem/ShiftAddLLM$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 generate_text/generate_llama.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --load BCQ_ACC_Llama-3.1-8B-Instruct --benchmark
Model: meta-llama/Meta-Llama-3.1-8B-Instruct
Device: cuda
Max Length: 300
Top-k: 0
Temperature: 0.7
Do Sampling: True
========================================
==== Benchmarking unquantized model ====
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 17.26it/s]
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Generated Text (Unquantized Model):
Once upon a time, in the land of music, there was a magical instrument called the "Soundwave". It was a beautiful, shimmering harp with strings that sang in harmony with the wind and the stars. The Soundwave was said to have the power to bring people together, to heal the sick, and to make the world a more harmonious place.
One day, a young musician named Luna stumbled upon the Soundwave in a hidden glade deep in the forest. She had been searching for it her whole life, and when she finally found it, she felt a surge of excitement and joy.
As soon as Luna touched the strings of the Soundwave, she felt a deep connection to the instrument. She began to play, and the music that flowed from her fingers was like nothing anyone had ever heard before. It was as if the Soundwave was singing through her, using her as a vessel to express its beauty and magic.
As Luna played, the forest around her began to transform. The trees swayed to the rhythm, the flowers bloomed in time with the melody, and the creatures of the forest danced and sang along. The music was so powerful that it seemed to bring the very essence of the forest to life.
Word of the Soundwave's magical music spread quickly, and soon people from all over the land came to hear Luna play. They would sit in the glade, mesmerized by the beauty and harmony of the music, and feel their hearts and
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Unquantized Model):
In a galaxy far, far away, a brave young space pilot named Nova sets out to explore the cosmos and save the galaxy from an ancient evil. With her trusty spaceship, the Nova Star, and her loyal AI sidekick, Zeta, Nova embarks on a thrilling adventure that takes her to the farthest reaches of the galaxy.
From the desolate wastelands of the Dark Nebula to the lush, vibrant jungles of the Orion Cluster, Nova and Zeta encounter strange and fantastical worlds, alien species, and hidden dangers at every turn. Along the way, they must navigate treacherous asteroid fields, hostile alien encounters, and the ever-present threat of the dark forces that seek to destroy the galaxy.
As Nova and Zeta journey deeper into the unknown, they discover that the fate of the galaxy hangs in the balance. An ancient evil, known only as the "Devourer," threatens to consume all in its path, leaving nothing but destruction and despair in its wake. Nova and Zeta must use all their skills and cunning to defeat the Devourer and save the galaxy from annihilation.
Will Nova and Zeta be able to overcome the odds and save the galaxy, or will the darkness consume them all? Join Nova on her epic adventure through the cosmos and discover the wonders and dangers that await her in the galaxy far, far away.
In this thrilling space opera, Nova and Zeta face off against formidable foes, uncover hidden secrets, and discover
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Unquantized Model):
The quick brown fox jumps over the lazy dog. This is a sentence that is commonly used as a demonstration of a typographical arrangement of keys on a standard English keyboard layout. It is often used to test the keyboard layout. This sentence is a pangram, a sentence that uses all the letters of the alphabet at least once. It is a classic example of a pangram and is widely known and used. The sentence is often used in typing drills and keyboard testing programs. It is a useful tool for testing the layout and functionality of a keyboard.
The sentence is also often used in word games and puzzles, such as crosswords and word searches, because it uses all the letters of the alphabet. It is a challenging sentence to type quickly and accurately, because it requires the typist to use all the keys on the keyboard in a specific order. This makes it a useful tool for testing typing skills and improving typing speed and accuracy.
The sentence has become a well-known phrase and is often referenced in popular culture. It has been used in a variety of contexts, including music, film, and literature. It is a phrase that is often used to represent the idea of a challenge or a test of skill.
In addition to its use in typing drills and word games, the sentence has also been used in a variety of other contexts. It has been used in advertising and marketing, as a way to test the layout and functionality of a keyboard. It has also been used in educational settings, as
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Unquantized Model):
In the middle of the night, she heard a sound that made her heart skip a beat. It was a faint scratching noise coming from the closet. She tried to brush it off as the house settling, but the noise persisted. She slowly got out of bed and approached the closet, her heart racing with anticipation. As she opened the door, a faint light flickered in the darkness. Suddenly, a figure emerged from the shadows, its eyes glowing with an otherworldly intensity. She froze, paralyzed with fear, as the figure began to speak in a low, raspy voice. "I've been waiting for you." The voice sent shivers down her spine, and she tried to scream, but her voice was trapped in her throat. The figure began to move closer, its presence filling the room with an unspeakable horror. She was trapped, unable to move or escape. The figure's eyes seemed to be drawing her in, pulling her into a dark and foreboding world. She felt herself being pulled towards it, helpless to resist. The last thing she remembered was the feeling of cold breath on her skin, and then everything went black. When she came to, she was lying on her bed, the room bathed in the warm light of dawn. It was just a dream, she told herself, trying to shake off the feeling of unease. But as she looked around the room, she noticed something strange. The closet door was open, and on the
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Unquantized Model):
The world was on the brink of change, and the wind was blowing strong. The year was 1989, and the Berlin Wall, a physical and symbolic barrier between East and West, was about to come tumbling down. It was a time of great hope and uncertainty, as the people of Europe and the world watched with bated breath as the once-impenetrable wall was breached.
It was against this backdrop that a small group of visionaries, led by the charismatic and visionary Dr. Roger Williams, began to imagine a new kind of university – one that would bring together students, faculty, and staff from all over the world to learn, grow, and thrive in a truly global community.
In 1991, the University of the Nations (UofN) was born, with its first campus in Kona, Hawaii. From the outset, UofN was designed to be a place where people from diverse backgrounds and cultures could come together to learn, share, and grow in their faith, their character, and their callings.
Today, UofN is a global network of campuses and training centers in over 180 countries, with thousands of students, faculty, and staff from all walks of life. From its humble beginnings, UofN has grown into a vibrant and diverse community of learners, leaders, and change-makers who are making a difference in their worlds and in the world at large.
At UofN, we believe that education is not just about
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Unquantized Model):
It all started with a simple mistake. A misplaced decimal point, a misread temperature reading, and a resulting error that would shake the very foundations of the scientific community. It was 2012, and the discovery of the Higgs boson, a fundamental particle predicted by the Standard Model of particle physics, was on the horizon.
But what if the mistake wasn’t a mistake at all? What if it was a clever ruse, a deliberate attempt to mislead the scientific community and cover up a far more extraordinary truth?
Enter the enigmatic physicist, Dr. Sophia Patel. A brilliant mind with a reputation for being one step ahead of the curve, Sophia has been quietly working on a top-secret project, one that could change the course of human understanding forever.
As the world celebrates the discovery of the Higgs boson, Sophia knows that something is off. The data doesn’t add up, and she suspects that her colleagues may be hiding something from her. With the help of a young and ambitious journalist, Alex Chen, Sophia embarks on a perilous journey to uncover the truth.
Their investigation takes them from the highest echelons of academia to the darkest corners of the scientific underworld, where the stakes are high and the players are ruthless. As they dig deeper, they begin to unravel a web of deceit and conspiracy that threatens to upend everything they thought they knew about the universe.
The Higgs boson may be a discovery, but it’s just the tip of the
Tokens per second (Unquantized Model): 30.37 tokens/second
Tokens per second (Unquantized Model): 34.86 tokens/second
Tokens per second (Unquantized Model): 35.27 tokens/second
Tokens per second (Unquantized Model): 35.32 tokens/second
Tokens per second (Unquantized Model): 35.10 tokens/second
Tokens per second (Unquantized Model): 35.04 tokens/second
Avg TPS (Unquantized Model): 34.33 tokens/second
========================================

===== Running quantized model =====
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 13.56it/s]
/home/ubuntu/llm-train-filesystem/ShiftAddLLM/generate_text/generate_llama.py:93: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(model_checkpoint))
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Quantized Model):
Once upon a time, in the bustling streets of Tokyo, there existed a small, mysterious shop called "Moonlit." It was an unassuming place, tucked away in a quiet alley, with a faded sign above the door that read "Moonlit" in delicate, cursive script. The shop itself was an enigma, even to the locals, who would often whisper about the strange, otherworldly items they'd seen being carried out of its doors.
Inside, Moonlit was a labyrinth of narrow corridors, dimly lit by candles and lanterns, with shelves upon shelves of curiosities and trinkets that seemed to shimmer and shine with an ethereal light. Some said the shop was run by a family of skilled spiritualists, who possessed an uncanny ability to sense the energies of the items they sold. Others whispered that the proprietor, a quiet, enigmatic woman with piercing green eyes, was a medium, who communed with the spirits of the dead to know the true nature of the artifacts.
Those who ventured into Moonlit often reported feeling a sense of disorientation, as if the very fabric of reality had been tweaked, ever so slightly, by their presence in the shop. Others said it was a place of wonder, where the boundaries between worlds were blurred, and the impossible became possible.
One day, a young woman named Akane wandered into Moonlit, searching for a gift for her dying grandmother. She had heard whispers of a rare, exquisite orch
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Quantized Model):
In a galaxy far, far away, there lived a star named Sirius. This star, located in the constellation Canis Major, was known for its incredible brightness and was considered one of the brightest stars in the sky. However, as time went by, Sirius's brightness began to fade, and the star was eventually deemed a "dwarf" star, meaning it was much smaller than other stars in the galaxy.

One day, a young astronomer named Sarah stumbled upon Sirius's fading brightness while studying the galaxy. Sarah was fascinated by Sirius's descent into darkness and decided to conduct a research on the star's evolution. Sarah spent countless hours studying Sirius's brightness, observing its changes over time, and eventually published a paper on the star's decline.

However, as Sarah delved deeper into the galaxy, she began to notice that other stars, too, were fading at an incredible rate. She realized that the galaxy's collective brightness was actually decreasing, not just Sirius's. This realizati[0/547]ked a new line of research, which led Sarah to conclude that the galaxy was experiencing a galactic-wide "darkening" phenomenon. The study of this phenomenon, known as "galactic darkness," led Sarah to discover that the galaxy's collective brightness was actually being influenced by an external force.

Sarah, being the young astronomer that she was, decided to investigate this phenomenon further. She spent countless hours analyzing the galaxy's collective brightness, observing its patterns, and trying to understand the external force that was causing
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Quantized Model):
The quick brown fox jumps over the lazy dog.
This sentence is often used to demonstrate a sentence that uses all 26 of the letters in the English alphabet. It is a bit of a trick, though, as "jumps" and "dog" don't actually use the letters "j", "x", or "q", which are not part of the standa
rd 26 letters. However, if you remove the three letters that are not part of the standard 26 letters, you are left with "The quick brown fox". This sentence uses all 23 of the standard letters. A quick fox jumps over the lazy dog. A quick fox ju
mps over the lazy dog. A quick fox jumps over the lazy dog. A quick fox jumps over the lazy dog. A quick fox jumps over the lazy dog. A quick fox jumps over the lazy dog. A quick fox jumps over the lazy dog. A quick fox jumps over the lazy dog.
A quick fox jumps over the lazy dog. A quick fox jumps over the lazy dog. A quick fox jumps over the lazy dog. A quick fox jumps over the lazy dog. A quick fox jumps over the lazy dog. A quick fox jumps over the lazy dog. A quick fox jumps over
the lazy dog. A quick fox jumps over the lazy dog. A quick fox jumps over the lazy dog. A quick fox jumps over the lazy dog. A quick fox jumps over the lazy dog. A quick fox jumps over the lazy dog. A
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Quantized Model):
In the middle of the night, she heard a sound that made her jump. It was the sound of her daughter, Emily, stirring in her sleep. The 3-year-old girl was crying out in her sleep, but she was also having a nightmare, so the sound she was making w
as not just a cry, but a loud, jarring sound that seemed to shake the very foundations of the house.
As she listened to her daughter's cries, the mother felt a wave of fear wash over her. She knew that Emily was having a nightmare, and that she was scared. But she also knew that she had to get up and check on her daughter, because she was still
 very young, and she might need her.
The mother carefully got out of bed, not wanting to scare her daughter any more. She walked over to Emily's bed, where she was lying on her side, her eyes closed, her small face twisted up in a cry. The mother gently reached down and picked up h
er daughter, cradling her in her arms, and trying to comfort her.
"Shh, baby," she said, trying to calm her down. "It's okay. Mommy is here. You're safe."
The mother held Emily close, trying to soothe her. She talked softly to her, trying to calm her down. She stroked her small face, trying to comfort her. But it was clear that Emily was not going to calm down. She was still crying
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Quantized Model):
The world was on the brink of change. The Industrial Revolution had sparked innovation, efficiency, and progress. Cities were growing, and with them, the demand for housing was increasing. This created opportunities for entrepreneurs to build af
fordable housing for workers, which they dubbed "workers' housing." This housing, built by entrepreneurs, was typically small, functional, and simple in design. It provided a place for workers to live, but the quality of life was limited. The wo
rkers were not happy with their housing, as it lacked comfort and amenities. They yearned for a better life. This tension between the desire for workers' housing and the lack of amenities would continue throughout history. As society evolved, ho
using became a major issue. The Industrial Revolution sparked innovation, but it also created social and economic challenges. The gap between workers' housing and the quality of life they desired widened. The need for housing that met workers' n
eeds, with amenities, grew. This led to innovations in construction, materials, and technology, which, in turn, created new opportunities for entrepreneurs to build housing for workers. The workers' housing phenomenon continued to evolve through
out history, with innovations in construction, materials, and technology creating new opportunities for entrepreneurs to build housing for workers. This tension between the desire for workers' housing and the lack of amenities continued througho
ut history. The world was on the brink of change. The Industrial Revolution had sparked innovation, efficiency, and progress. Cities were growing, and with them, the demand for housing
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Quantized Model):
It all started with a simple mistake. I had meant to buy a book on baking, but I ended up buying a book on how to cook. And not just any book on how to cook, but a book with a title that read "The Joy of Cooking".
I had been looking for a cookbook for my mom, and I had decided to get her a book that would help her with her cooking. But, in my haste, I had ended up buying a book that was meant for a more experienced cook. The book had a title that read "Th
e Joy of Cooking", and it was a big book, with lots of pages and a lot of information.
I had taken it home with me, and I had been meaning to give it to my mom. But, as I was reading through it, I had realized that it was a book that was meant for a more experienced cook. The book had a lot of recipes, but it was a book that was m
eant for a person who was already experienced in cooking.
So, I decided to keep the book for myself, and to use it to learn more about cooking. And, as I was reading through it, I realized that it was a book that was written in a way that was easy to understand, even for a person who was new to cooking
.
The book had a lot of recipes, and it had a lot of information on different types of cooking, such as baking, roasting, and grilling. And,
Tokens per second (Quantized Model): 35.14 tokens/second
Tokens per second (Quantized Model): 35.02 tokens/second
Tokens per second (Quantized Model): 35.36 tokens/second
Tokens per second (Quantized Model): 35.69 tokens/second
Tokens per second (Quantized Model): 35.69 tokens/second
Tokens per second (Quantized Model): 35.26 tokens/second
Avg TPS (Quantized Model): 35.36 tokens/second
===================================
```

3. BCQ_LAT_Llama-3.1-8B-Instruct.pt

```
ubuntu@150-136-66-218:~/llm-train-filesystem/ShiftAddLLM$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 generate_text/generate_llama.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --load BCQ_LAT_Llama-3.1-8B-Instruct.pt --benchmark
Model: meta-llama/Meta-Llama-3.1-8B-Instruct
Device: cuda
Max Length: 300
Top-k: 0
Temperature: 1.0
Do Sampling: True
========================================
==== Benchmarking unquantized model ====
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 17.38it/s]
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Generated Text (Unquantized Model):
Once upon a time, in the land of music, there was a magical instrument called the Crystal Sonata. It was said to be the most beautiful and enchanting instrument in all the land, capable of playing the most mesmerizing melodies and harmonies that could entrance even the most skeptical of hearts.
One day, a young musician named Lily heard the legend of the Crystal Sonata and became determined to find it. She traveled far and wide, searching for the instrument, meeting many people along the way who claimed to have seen or heard it. But no matter how hard she searched, Lily couldn’t seem to find it.
Just when she was about to give up, Lily stumbled upon a small, mysterious shop tucked away in a quiet alley. The sign above the door read “Magical Instruments” and the windows were filled with an assortment of strange and exotic instruments. Lily’s heart skipped a beat as she pushed open the door and stepped inside.
The shop was dimly lit, but Lily’s eyes quickly adjusted to the soft glow of the lanterns that hung from the ceiling. She wandered through the aisles, running her fingers over the intricate carvings on the instruments, feeling an energy emanating from each one. As she turned a corner, she caught sight of a pedestal in the center of the room, and on top of it, the Crystal Sonata.
It was even more beautiful than she had imagined. The crystal keys sparkled like diamonds in the light, and the body of the
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Unquantized Model):
In a galaxy far, far away, 3DCG will find a way. Robots have evolved, systems have grown, and now the technology is at a point where 3DCG animation can not only be made at home, but also be produced with professional-quality polish. In this tutorial, we’ll show you how to create an animated light cycle from the movie Tron Legacy.
In a galaxy far, far away, 3DCG will find a way. Robots have evolved, systems have grown, and now the technology is at a point where 3DCG animation can not only be made at home, but also be produced with professional-quality polish. In this tutorial, we’ll show you how to create an animated light cycle from the movie Tron Legacy. First, you’ll need to have some experience in 3D modeling and animation, as well as some software like Blender or Autodesk Maya. We’ll be using Blender for this tutorial, as it’s free and open-source.
Step 1: Set up your software and modeling space
Create a new project in Blender by going to the File menu and selecting New. You should then set your render engine to Cycles, as it’s the most accurate and physically based renderer in Blender.
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Quantized Model):
The quick brown fox jumps over the lazy dog.
There are 26 letters in the sentence "The quick brown fox jumps over the lazy dog."
with numbers, and they can sometimes be unpredictable.
This is a special keyboard character that means a shortcut key on the keyboard. Most keyboards have the following two keys.
Use the shortcuts listed below to follow along.
a = clear
alt =  Alt key  (This key can also change some keys that are held down)
shift = Shift key
Windows = A keyboard key commonly referred to as the "Windows" key (This is NOT a keyboard key)
Enter = The Enter key. This is also called the "Return" key.
" {} | — These are called the "curly" or "backets" (Note: These keys do not act like any other keys on the keyboard. The {} are always inserted and can be erased by following a described system.)
" [, ] { } = The same < > * *<>%reset insert nothing}
        " {<}}.reset"*= ()* -= The Equals key on the keyboard acts like the modulus operator (like the % in languages like Mathematica). To see the special use of the modulus operator goto Mathematica website The Modulus key on the keyboard acts like the modulus operator (like the % in languages like Mathematica). Going to Mathematica website

The modulus operator does nothing when pressed but it does insert a ^ well by other keys or characters
This web
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Quantized Model):
In the middle of the night, she heard a sound - something metallic scraping against the kitchen counter. As she turned to face the noise, a face appeared - inches from hers.
A native of the Philippines, Angela Sy Cemerong, age 22, had been working in New York for three months when the incident happened. 'It was around 2 or 3 in the morning. I was about to go to sleep,' she recalls. 'I felt something touching the counter. At first, I didn't think anything of it. But then, I realized that I had heard a rustling sound, and I got a little scared. I figured that someone had just pushed something across the counter by accident. But then I saw the thing's face staring back at me. That's when I screamed. I screamed so loud that I woke up my roommate. I told her what happened, and she asked if I was okay, because I was shaken. My room mate told me not to worry, that it was probably just a one or two bad dreams she was having that night. She said that I should be grateful, that I had gotten a clean slate in my life, that things were moving up for me. I thanked her. That was five months ago. There's no indication that anything abnormal has happened since, and I do not sense any doom oring of substance.
Yet, Angela, thought happen after she turned off the lamp and waited for what would come next -
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Quantized Model):
The world was on the brink of change, with many people thinking it was impossible for world peace to come to fruition. However, in 1982, at a time when war was common and fear was the commonality, Helena Bonham Carter, an actress in the British film "National Velvet," went to Uganda to help three orphaned elephants. This act of kindness evolved into the helping of thousands, while the world was evolving towards peaceful existence. Carol Stangel asked: 'What was so powerful about the elephants?  This global act of kindness helped end war in many places of the world. It is of such significant implications, that... if any human part of Bonham-Carter arrived in Uganda, he or she would have been arrested. The scale of global peace was realized, right on that flight from New York, USA, with Ugandan use of 'Bindi" which Carol utilized in Kenya to show Uganda-US relation locale visitors arriving Uganda Uganda disbalance lasted between Kenya Uganda unde Ugnda Kenya Uganda USA 509500 Tut hunters hunter JHUrev Uganda Kenya UA documented surveyed hunters Kenya Kenya Uganda Uganda Kenya Kenya ug Uganda Uganda Kenya camp Kenya suppl ug Uganda hunter vill Ken Uganda man Kenya huntmen UA Kenya her Carm Uganda hunted K Ug Kenya tug Ashington Kenya hunt Ug Uganda hunt Ug Uganda hunt Uganda Kenya hunt ug Ug Uganda hunt son ug Uganda Uganda SA capt Kenya scare Uganda Ltd Uk http Ug Uganda Local Eszcze Agricultural hunt ugld Uganda Uganda reign savvy hunt ug Italy Rwanda ug
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Text (Quantized Model):
It all started with a simple mistake - or so I thought. I had been meaning to fix a showerhead, which had somehow come loose and was spraying water everywhere. (Why does this always happen at the worst possible moment?)
I called in a handyman, who shortly after arriving started explaining why he thought he should get a raise. Why, he asked, did management pay him to fix a showerhead? Which led to a long conversation (in excrucab).
When you encounter language as pretty as Macana Airport's menu (ingredients)"
walk silently behind spectators packed At Large Variety airport).
OnlywhenS how engineer ply plates groceries waters oatmeal+dians important personal virtue+
person tomato discussion *
Tsutton Darwin trim Pel Fish0503061806659969959969959pal9892939969969959969956969969959969959959969959959969956p6159956956959969956969959969956rut995695999996996995996995695995696995695995696995696995695995696995695995996995695995696995696995696995696995696995696995695995696995696995695995695995696995695696995695996995696995696995696995695995695996995696995696995696995695995696995696995696995696995696995696995696995995696995695996995695995696995696995696995696995696995696
Tokens per second (Quantized Model): 35.31 tokens/second
Tokens per second (Quantized Model): 35.38 tokens/second
Tokens per second (Quantized Model): 35.77 tokens/second
Tokens per second (Quantized Model): 36.26 tokens/second
Tokens per second (Quantized Model): 35.81 tokens/second
Tokens per second (Quantized Model): 35.61 tokens/second
Avg TPS (Quantized Model): 35.69 tokens/second
```