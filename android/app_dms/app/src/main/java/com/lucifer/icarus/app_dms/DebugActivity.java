package com.lucifer.icarus.app_dms;

import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.lucifer.icarus.app_dms.GlobalObject;
import com.lucifer.icarus.app_dms.ICAApplication;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class DebugActivity extends AppCompatActivity {

    private TextView mqttipTV;
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_debug);

        mqttipTV = (TextView)findViewById(R.id.editText_mqttip);
        mqttipTV.setText(GlobalObject.getInstance().getMqttIP());


        ((Button)findViewById(R.id.button_mqttip_commit)).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String s = mqttipTV.getText().toString();

                String ip = "^(1\\d{2}|2[0-4]\\d|25[0-5]|[1-9]\\d|[1-9])\\."
                        +"(1\\d{2}|2[0-4]\\d|25[0-5]|[1-9]\\d|\\d)\\."
                        +"(1\\d{2}|2[0-4]\\d|25[0-5]|[1-9]\\d|\\d)\\."
                        +"(1\\d{2}|2[0-4]\\d|25[0-5]|[1-9]\\d|\\d)$";//限定输入格式
                Pattern p=  Pattern.compile(ip);
                Matcher m=p.matcher(s);
                boolean b=m.matches();

                if(b){
                    GlobalObject.getInstance().setMqttIP(s);
                    ICAApplication.getInstance().exit();
                }else{
                    Toast.makeText(DebugActivity.this, "ip输入格式有误，请重新输入", Toast.LENGTH_SHORT).show();
                }


            }
        });

        ((Button)findViewById(R.id.button_mqttip_cancel)).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish();
            }
        });

        ((Button)findViewById(R.id.button_cancel)).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish();
            }
        });
    }
}

