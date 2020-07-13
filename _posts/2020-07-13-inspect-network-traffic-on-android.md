---
layout: post
title: Inspect network traffic on Android
---

As an Android developer, you might want to inspect network traffic in your Android device. With the apps you build, in fact, there are a lot of tools help you debug HTTP(s) traffic. However, how can you monitor other various apps such as: Youtube, Twitter, etc.?
This article will help you do it by using mitmproxy tool.

<div class="message">
This post also published on <a href="https://medium.com/@thuat26/inspect-network-traffic-on-android-ce077f6ec867">my Medium blog</a> .
</div>

## mitmproxy
In short, mitmproxy is an interactive man-in-the-middle proxy for HTTP and HTTPS with a console interface, and most importantly it's free!
You can read <a href="https://docs.mitmproxy.org/stable/">this document</a>  to understand how it works.

## Prerequisites
- Mitmproxy tool
- Android Emulator with root permission

<div class="message">
Important Note: When you create the emulator, you must choose "(Google APIs)" in the Target (android version), do not choose "(Google Play)" or you will not be able to get adb root access.
</div>

## Idea
Ideally, we install [the mitmproxy CA certificate manually](https://docs.mitmproxy.org/stable/concepts-certificates/#installing-the-mitmproxy-ca-certificate-manually) as a user-added CA and done!

Unfortunately, [since Android 7, apps ignore user-added CAs](https://android-developers.googleblog.com/2016/07/changes-to-trusted-certificate.html), unless they are configured to use them. And most applications do not explicitly opt in to use user certificates. So, we need to place our mitmproxy CA certificate in the system certificate store as a trusted CA.
Now let's start!

## Create mitmproxy certificate
### Install mitmproxy
{% highlight text %}
brew install mitmproxy
{% endhighlight %}

### Generate certificate

{% highlight text %}
mitmproxy
{% endhighlight %}

### Rename certificate
- Enter your certificate folder

{% highlight text %}
cd ~/.mitmproxy/
{% endhighlight %}

- CA Certificates in Android are stored by the name of their hash, with a '0' as extension. Now generate the hash of your certificate
{% highlight text %}
openssl x509 -inform PEM -subject_hash_old -in mitmproxy-ca-cert.cer | head -1
{% endhighlight %}

- For example, the output is `your_hash_value`. We can now copy `mitmproxy-ca-cert.cer` to `your_hash_value.0` and our system certificate is ready to use
{% highlight text %}
cd ~/.mitmproxy/
cp mitmproxy-ca-cert.cer your_hash_value.0
{% endhighlight %}

### Insert certificate into system certificate store
- Enter emulator folder within Android SDK
{% highlight text %}
cd .../Android/SDK/emulator/
{% endhighlight %}

- Get a list of your AVDs with emulator -list-avds
{% highlight text %}
./emulator -list-avds
{% endhighlight %}

- Start your android emulator with `-writable-system` option in order to write to `/system`
{% highlight text %}
./emulator -avd <avd_name_here> -writable-system
{% endhighlight %}

- Restart adb as root
{% highlight text %}
adb root
{% endhighlight %}

- Remount the system partition as writable
{% highlight text %}
adb shell "mount -o rw,remount /"
{% endhighlight %}

- Push your certificate to the system certificate store and set file permissions
{% highlight text %}
adb push your_hash_value.0 /system/etc/security/cacerts
adb shell "chmod 664 /system/etc/security/cacerts/your_hash_value.0"
{% endhighlight %}

- Reboot your emulator
{% highlight text %}
adb reboot
{% endhighlight %}

Now we installed the CA certificate on Emulator.

## Setup Proxy on Emulator
Open Emulator Settings, add manual proxy with hostname: `127.0.0.1` and port `8080`

Launch the tool and see the magic!
You can start any of three tools from the terminal:
- [mitmproxy](https://docs.mitmproxy.org/stable/tools-mitmproxy/) -> gives you an interactive TUI
- [mitmdump](https://docs.mitmproxy.org/stable/tools-mitmdump/) -> gives you a plain and simple terminal output
- [mitmweb](https://docs.mitmproxy.org/stable/tools-mitmweb/) -> gives you a browser-based GUI

For instance, I open the Youtube app and monitor the traffic as below.
- Run `mitmproxy`
<p align="center">
<img src="/assets/mitmproxy.png"/>
</p>

- Run `mitmweb` to see the API details
<p align="center">
<img src="/assets/mitmweb.png"/>
</p>

Voila!