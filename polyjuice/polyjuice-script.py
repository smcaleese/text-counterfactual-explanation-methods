from polyjuice import Polyjuice

def main():
    pj = Polyjuice(model_path="uw-hai/polyjuice", is_cuda=True)

    # text = "On a dark, gloomy New Year's Eve night, an ill nurse, her life slowly ebbing away, demands that David Holm be presented to her at once. We don't yet know who David Holm is, or why this nurse wishes to see him, but her only dying wish is to speak with him just one more time. On the other side of the town, nestled comfortably amongst the gravestones of the local cemetery, Holm (Victor Sjöström, who also directed) and two of his drunken associates merrily await the coming of the New Year. Here we can tell just when to drink"
    # text = "I liked the movie."
    # text = "I would put this at the top of my list of films in the category of unwatchable trash! There are films that are bad, but the worst kind are the ones that are unwatchable but you are suppose to like them because they are supposed to be good for you! The sex sequences, so shocking in its day, couldn't even arouse a rabbit. The so called controversial politics is strictly high school sophomore amateur night Marxism. The film is self-consciously arty in the worst sense of the term. The photography is in a harsh grainy black and white. Some scenes are out of focus or taken from the wrong angle. Even the sound is bad! And some people call this art?"
    # text = "I really hated this movie."
    # text = "This movie was sadly under-promoted but proved to be truly exceptional. Entering the theatre I knew nothing about the film except that a friend wanted to see it. I was caught off guard with the high quality of the film. I couldn't image Ashton Kutcher in a serious role, but his performance truly exemplified his character. This movie is exceptional and deserves our monetary support, unlike so many other movies. It does not come lightly for me to recommend any movie, but in this case I highly recommend that everyone see it. This films is Truly Exceptional!"
    text = "On a dark, gloomy New Year's Eve night, an ill nurse, her life slowly ebbing away, demands that David Holm be presented to her at once. We don't yet know who David Holm is, or why this nurse wishes to see him, but her only dying wish is to speak with him just one more time. On the other side of the town, nestled comfortably amongst the gravestones of the local cemetery, Holm (Victor Sjöström, who also directed) and two of his drunken associates merrily await the coming of the New Year. Here we can tell just when to drink"

    perturbations = pj.perturb(
        orig_sent=text,
        ctrl_code="negation",
        num_perturbations=1,
        perplex_thred=None,
        verbose=True
    )
    print("perturbations:", perturbations)

if __name__ == "__main__":
    main()
