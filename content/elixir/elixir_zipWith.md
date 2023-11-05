---
title: Utilizing the ```Enum.zip_with/3``` Function
date: 2023-08-16
tags: elixir, haskell
group: elixir
order: 2
--- 

---

One thing that is worth mentioning for people who might find themselves going back and forth with projects, or even languages, is the use of Enum when you might just think to reach for the List module. The Enum module provides many useful methods, but I just always find myself looking for them first in the List module. Because lists are implemented as linked data structures, they’re good for recursion, but bad for randomly retrieving an element or even figuring out the length because you’d need to traverse the whole list to figure out the size. 

---


I think I finally accepted the Enum module superiority though when I finally came across the [zip_with](https://hexdocs.pm/elixir/1.12/Enum.html#zip_with/3). At this point in my life I think I am actively avoiding anything Haskell, but I guess we can give this one a pass. This is a common method used in Haskell and it has also proven useful to me in Elixir. I am usually finding myself, for whatever reason, with equal length lists depicting various aspects of the system In a pairwise fashion. For example, when gathering data for users to compile into some useful statistics, I can neatly separate them into different lists, and then we can perform operations on them as so:


---

I have gone back and forth enough times now to finally realize there are times when you may need it in both forms. At first, I’d just gather the data as [name, score, id], [name, score, id] … and we can simply map over that. But there are other times when having the lists separate is helpful (especially for display purposes). In that case, we gather the data in this form and we have three separate lists of ```[name, name …]```, ```[score, score …]```. ```[id, id …]``` – each of equal length.

---
From the Elixir docs:
>It's important to remember that zipping inherently relies on order. If you zip two lists you get the element at the index from each list in turn.


It is also worth noting that there are other considerations here, too. For example, having the data in a list does not allow us to access them by key etc.. Just another case of “you need to carefully plan out your data structures”.
In this case I just have separate lists of strings which likely won’t illustrate most of the useful of the feature, but even here it can make things a little easier for us. One pretty common thing to do is to alter the display based on certain criteria – let’s say that if it is the person themselves in that list, we want to highlight that. Rather than going back and having to restructure our query to get all that user data in one row, we can perform a zip_with operation and achieve the same result. It might seem like a rather contrived example but I have found myself here a few times before 😊.

---

```elixir
  prop leader_names, :list
  prop leader_scores, :list
  prop leader_ids, :list
```


```elixir
    leader_names = Enum.map(leader_list, fn item -> List.first(item) end)
    leader_scores = Enum.map(leader_list, fn item -> List.last(item) end)
    leader_ids = Enum.map(leader_list, fn item -> List.pop_at(item, 1) |> Kernel.elem(0) end)

    zipped = Enum.zip_with(leader_ids, leader_scores, fn x,y -> '#{x}--#{y}' end)
```

```shell-session
zipped: ['1--2200', '5--2200', '9--2200', '2--2150', '13--2150', '6--1950', '10--1950', '14--1450', '11--1350', '7--1350']
```
