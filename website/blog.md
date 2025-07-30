---
layout: default
title: Hyperblog
permalink: /blog/
---

# The Hyperblog

<a href="{{ "/feed.xml" | relative_url }}" class="subscribe-button">Subscribe via RSS</a>

<ul>
  {% for post in site.posts %}
    <li>
      <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
      <p>{{ post.excerpt }}</p>
      <p><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%B %d, %Y" }}</time></p>
    </li>
  {% endfor %}
</ul>