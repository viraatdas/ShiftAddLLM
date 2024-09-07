## Usage
1. Quantize model
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 model/llama.py meta-llama/Llama-2-13b-hf --wbits 3 --groupsize 128 --acc --bcq --bcq_round 50 --load BCQ_ACC_Llama-2-13b-hf # bcq_round 20 works too, bigger - slower - maybe better
```

2. Generate
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 generate_text/generate_llama.py --model meta-llama/Llama-2-13b-hf --load BCQ_ACC_13b_HF --benchmark
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

