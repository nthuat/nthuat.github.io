---
layout: post
title: How to save file to external storage in Android 10 and Above
---

Since Android 10, [Environment.getExternalStoragePublicDirectory](https://developer.android.com/reference/kotlin/android/os/Environment?hl=en#getexternalstoragepublicdirectory) method was deprecated, then you was unable access public external storage directory directly to save your file. This short post will give a solution to solve it.

As a user, I would like to download a file (image, pdf, etc.) from Internet and save to public Downloads folder in device, then the code below works like a charm in Android target API 28.

```kotlin
fun saveFileToExternalStorage(url: String, fileName: String): Completable {
    return Completable.fromCallable {
        val target = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
            fileName
        )
        URL(url).openStream().use { input ->
            FileOutputStream(target).use { output ->
                input.copyTo(output)
            }
        }
    }
}
```

However, if you run it in Android API 29+ (Android 10 and above), it will throw the exception:

```
java.io.FileNotFoundException: /storage/emulated/0/Download/your_file_name: open failed: EACCES (Permission denied)
```

Fortunately, we can achieve the same result by using [MediaStore](https://developer.android.com/reference/kotlin/android/provider/MediaStore.html?hl=en) approach.

```kotlin
@RequiresApi(Build.VERSION_CODES.Q)
private fun saveFileUsingMediaStore(context: Context, url: String, fileName: String) {
    val contentValues = ContentValues().apply {
        put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
        put(MediaStore.MediaColumns.MIME_TYPE, your_mime_type)
        put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
    }
    val resolver = context.contentResolver
    val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, contentValues)
    if (uri != null) {
        URL(url).openStream().use { input ->
            resolver.openOutputStream(uri).use { output ->
                input.copyTo(output!!, DEFAULT_BUFFER_SIZE)
            }
        }
    } 
}
```

Note that you have to input right value of MIME_TYPE which depends on your file. Each file type have corresponding MIME type, for instance:

- .pdf -> application/pdf
- .png ->image/png
- .epub -> application/epub+zip

You can get more information about MIME types at [here.](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types)

Happy reading!