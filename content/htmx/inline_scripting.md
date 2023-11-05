---
title: Inline Scripting w/ Hyperscript (Controversial!!)
date: 2023-10-16
tags: htmx, rust
group: htmx
order: 1
--- 

---

One of the more interesting and still-unclear-to-me aspects of HTMX is hyperscript, its companion library that compares itself to jQuery. I only say still-unclear-to-me because at this point I am not really sure widespread its adoption will be. Anything that goes up against traditional JS in this space seems to have a steep climb, plus its syntax takes some getting used to. It’s nice, and its readable … but its just not what most people will be used to (at least from what I gather).

---


What I also learned along the way was that it is a controversial practice, which means sign me up. Per the docs...
> This is a controversial concept compared with the others listed here, and we consider it an “optional” rule for hypermedia-friendly scripting: worth considering but not required.


---

After using it for a little bit it does seem to carry a lot of power. I think what also goes into my still-unclear status also relates to just where it might fit into a typical project. Of course that could mean anything and it could be that start of the show or simply a little UI element used on some obscure page. I tend to think that the latter might be a little more suitable, because once we start getting past one or two mutations or reactions on a certain element, I can see things getting unwieldy, especially if you are someone who is not used to working with raw HTML a lot.

---

Here is some hyperscript that I am using to simply disables some form inputs based on the values of other elements. This is a pretty common scenario where we have a form and only certain fields make sense in certain cases, so we want to just go ahead and disable them to prevent the user from even trying to sneak something in. So, here, we have a first_name and last_name field and then a company_name field. Only one of these should ever be filled (first + last or company) so we can add that directly on our elements as such:

---

```html
      <li>
          <input 
            type="text" 
            name="client_company_name" 
            id="client_company_name"
            class="field-style field-full align-none" 
            placeholder="Client Company Name" 
            value="{{entity.client_company_name}}" 
            _="on keyup 
              if my.value 
              repeat for x in [#client_f_name, #client_l_name]
                add @readonly to x
              end
              otherwise
              repeat for x in [#client_f_name, #client_l_name]
                remove @readonly from x
              end" 
            />
      </li>
      <li>
          <input 
            type="text" 
            id="client_f_name" 
            name="client_f_name" 
            class="field-style field-split align-left" 
            placeholder="First Name" 
            value="{{entity.client_f_name}}" 
            _="on keyup 
              if my.value 
                add @readonly to #client_company_name
              otherwise
                if #client_l_name.value
                  add @readonly to #client_company_name
                otherwise
                  remove @readonly from #client_company_name"
            />
          <input 
            type="text" 
            id="client_l_name" 
            name="client_l_name" 
            class="field-style field-split align-right" 
            placeholder="Last Name" 
            value="{{entity.client_l_name}}" 
            _="on keyup 
              if my.value 
                add @readonly to #client_company_name
              otherwise
                if #client_f_name.value
                  add @readonly to #client_company_name
                otherwise
                  remove @readonly from #client_company_name"
          />
      </li>
```

---

So as you can see despite not being complicated at all and being pretty easy to read and understand, the formatting alone can result in some long spanning code that some people just might not prefer. I am still getting used to it myself, but in the end I think I will take the overall simplification that the library provides in the overall scheme of things and just learn to get used to having some larger HTML files (or template files in this case).

---

<h6>*As a side note, its always good to go back and understand these semantic HTML elements and what the attributes actually mean. It took me a couple tries to remember the exactly differences between disabled and readonly.</h6>