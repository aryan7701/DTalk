package com.example.dtalk.View_Pager

import android.app.Activity
import android.app.Application
import android.content.ContentResolver
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.drawable.BitmapDrawable
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.renderscript.Element
import android.util.Log
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import com.bumptech.glide.signature.ApplicationVersionSignature
import com.example.dtalk.R
import com.example.dtalk.ml.Pneumonia
import kotlinx.android.synthetic.main.activity_profile.*
import kotlinx.android.synthetic.main.fragment_pneumonia__detection_.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.schema.TensorType.FLOAT16
import org.tensorflow.lite.schema.TensorType.FLOAT32
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer


private const val ARG_PARAM1 = "param1"
private const val ARG_PARAM2 = "param2"


class Pneumonia_Detection_Fragment : Fragment() {
    // TODO: Rename and change types of parameters
    private var param1: String? = null
    private var param2: String? = null

    //Bringing image from gallery to the App
    private lateinit var imageURI: Uri
    private lateinit var bitmap:Bitmap


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            param1 = it.getString(ARG_PARAM1)
            param2 = it.getString(ARG_PARAM2)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?): View? {
        // Inflate the layout for this fragment

        val v =  inflater.inflate(R.layout.fragment_pneumonia__detection_, container, false)

        val imageDetect = v.findViewById<ImageView>(R.id.imageDetect)
        val selectButton = v.findViewById<Button>(R.id.select_button)
        val detectButton = v.findViewById<Button>(R.id.detect_button)
        val resultText = v.findViewById<TextView>(R.id.resultText)

        selectButton.setOnClickListener {

            val intent = Intent(Intent.ACTION_PICK)
            intent.type = "image/*"
            startActivityForResult(intent, 1)
        }




        return v
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        //for picking images
        if (requestCode == 1 && resultCode == Activity.RESULT_OK && data != null) {
            Log.d("OTP_", "Image Result Received")

            //geting the URI(location) of the image
            imageURI = data.data!!


            //setting the profile image selected in circular format
            imageDetect.setImageURI(imageURI)

            bitmap = MediaStore.Images.Media.getBitmap(context?.contentResolver,imageURI)

            //checking the ImageURI shouldn't be null and then calling the function uploadingImageinFireBaseStorage()
            detect_button.setOnClickListener {

                if (imageURI != null) {

//                    var resized: Bitmap = Bitmap.createScaledBitmap(bitmap, 150, 150, false)

                    var resized: Bitmap = Bitmap.createScaledBitmap(bitmap,150,150,true)


                    //performing input task for ML MODEL

                    val model = Pneumonia.newInstance(requireContext())


                    // Creates inputs for reference.
                    val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 150, 150, 1), DataType.UINT8)

                    val tbuffer = TensorImage.fromBitmap(resized)
                    val byteBuffer = tbuffer.buffer


                    Log.d("byte buffer shape", byteBuffer.toString())

                    Log.d("input shape", inputFeature0.buffer.toString())


                    inputFeature0.loadBuffer(byteBuffer)

                    // Runs model inference and gets result.
                    val outputs = model.process(inputFeature0)
                    val outputFeature0 = outputs.outputFeature0AsTensorBuffer


                    val max = getMaxOutput((outputFeature0.floatArray))


                    //display predicted value
                    resultText.text = outputFeature0.floatArray[1].toString()


                    // Releases model resources if no longer used.
                    model.close()


//                    SavingData(this, imageURI)

//                    Toast.makeText(this,"Loading", Toast.LENGTH_SHORT).show()

                }

            }

        }

    }

    //getting max probablity output
    fun getMaxOutput(arr: FloatArray): Int {

        var index = 0
        var min = 0.0f
        for (i in 0..40) {

            if (arr[i] > min) {
                index = i
                min = arr[i]
            }

        }
        return index

    }

    companion object {

        // TODO: Rename and change types and number of parameters
        @JvmStatic
        fun newInstance(param1: String, param2: String) =
            Pneumonia_Detection_Fragment().apply {
                arguments = Bundle().apply {
                    putString(ARG_PARAM1, param1)
                    putString(ARG_PARAM2, param2)
                }
            }
    }
}