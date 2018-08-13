---
layout: page
title: Tricks and Tips
permalink: /blog/
---


This is a place for me to stash some of the tricks and tips I use in my career. I might have a couple lines of pandas code to a more thorough sklearn or TensorFlow example. It might even be a mathematical explaination of an algorithm. Find the notebooks at my github! This is a work in progress and I am just starting this on September 12, 2018.

<ul class="listing">
{% for post in site.posts %}
  {% capture y %}{{post.date | date:"%Y"}}{% endcapture %}
  {% if year != y %}
    {% assign year = y %}
    <li class="listing-seperator">{{ y }}</li>
  {% endif %}
  <li class="listing-item">
    <time datetime="{{ post.date | date:"%Y-%m-%d" }}">{{ post.date | date:"%Y-%m-%d" }}</time>
    <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
  </li>
{% endfor %}
</ul>
